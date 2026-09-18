"""Orchestration tests for worker.hardsub_job, against a fake Supabase.

The OCR half cannot run here — no ffmpeg, no episode, maybe no GPU — and the
DeepSeek half must not: this is about what the job writes and in what order.
Which statuses are walked, whether junk lines are kept out of the cues table,
whether both .srt files are uploaded with the source one first, and above all
whether a requeued job reads the video again when it already has the lines.

`hardsub_ocr.run`, `translate_srt_deepseek.DeepSeek`, `.translate_cues` and
`.glossary_from_lines` are stubbed, so this needs no video, no model download
and no API key. The fake database is the one `test_worker_transcribe.py`
built; the two job kinds write the same tables in the same shape.

    python test_worker_hardsub.py
"""

import os
import sys

import hardsub_ocr as H
import translate_srt_deepseek as D
import worker as W
from test_worker_transcribe import make_db


# -------------------------------------------------------------------- fixture

# What hardsub_ocr.run returns for a three-line episode with one piece of scene
# text in the scan band: the car's number plate, read at 0.62 with no Chinese
# character in it — the exact shape the 凤凰特工 batch produced.
LINES = [
    {"index": 1, "start_ms": 1000.0, "end_ms": 2200.0, "text": "你怎么在这里", "confidence": 0.98},
    {"index": 2, "start_ms": 2500.0, "end_ms": 3100.0, "text": "GV-6606", "confidence": 0.62},
    {"index": 3, "start_ms": 3400.0, "end_ms": 5000.0, "text": "曼云 我们走", "confidence": 0.97},
    {"index": 4, "start_ms": 5200.0, "end_ms": 6400.0, "text": "别怕", "confidence": 0.71},
]


def ocr_result(lines=LINES):
    return {
        "lines": [dict(l) for l in lines], "cues": len(lines), "duration_s": 7.5,
        "scan_area_pct": [0.0, 70.0, 100.0, 19.0],
        "static_text_removed": ["Veo"], "settings": {"models": {"det": "v6-small"}},
        "timings": {"total_s": 12.3, "ocr_s": 8.0},
    }


class FakeDeepSeek:
    def __init__(self, api_key, model="deepseek-flash", **kw):
        self.model = model
        self.usage = {"prompt_tokens": 100, "completion_tokens": 50}


def install_stubs(monkey, *, calls, lines=LINES, translate=None, glossary=None):
    """Replace the OCR and API halves with recorded fakes."""

    def fake_run(video, output=None, **kw):
        calls["ocr"] = calls.get("ocr", 0) + 1
        calls["ocr_kwargs"] = kw
        return ocr_result(lines)

    def fake_glossary(texts, client, *, source, **kw):
        calls["glossary"] = calls.get("glossary", 0) + 1
        return glossary if glossary is not None else {"曼云": "Mạn Vân"}

    def fake_translate(cues, client, *, language, source, glossary, style, chunk,
                       log_prefix="", flush=None):
        calls["translate"] = calls.get("translate", 0) + 1
        calls["translate_glossary"] = glossary
        if translate is not None:
            return translate(cues, flush)
        out = {c.index: f"[vi] {c.text}" for c in cues}
        # Persisted per chunk in the real one; the worker relies on it.
        if flush:
            flush(dict(out))
        return out, []

    for obj, attr, fake in ((H, "run", fake_run), (D, "DeepSeek", FakeDeepSeek),
                            (D, "glossary_from_lines", fake_glossary),
                            (D, "translate_cues", fake_translate)):
        monkey.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, fake)


def new_job(db, **over):
    job = {"id": "job-1", "owner_id": "owner-1", "title": "Ep 1",
           "kind": "hardsub", "status": "queued",
           "video_path": "owner-1/ep1.mp4", "voice_id": None,
           "settings": {"language": "Vietnamese", "ocr_lang": "ch"},
           "total_cues": 0, "qc_summary": None}
    job.update(over)
    db.tables["jobs"].append(job)
    return job


def run_job(db, job, *, api_key="sk-test", **stub_kw):
    monkey, calls = [], {}
    install_stubs(monkey, calls=calls, **stub_kw)
    W._COLS.clear()
    W._beat.update(at=0.0, on=False)     # presence is not under test
    W._touch["at"] = 0.0
    had = os.environ.pop("DEEPSEEK_API_KEY", None)
    if api_key:
        os.environ["DEEPSEEK_API_KEY"] = api_key
    try:
        W.hardsub_job(db, job)
    finally:
        for obj, attr, orig in monkey:
            setattr(obj, attr, orig)
        os.environ.pop("DEEPSEEK_API_KEY", None)
        if had:
            os.environ["DEEPSEEK_API_KEY"] = had
    return calls


# ---------------------------------------------------------------------- tests

checks = []


def check(name):
    def wrap(fn):
        checks.append((name, fn))
        return fn
    return wrap


@check("a fresh job reads, translates, and uploads both .srt files")
def _():
    db = make_db()
    job = new_job(db)
    calls = run_job(db, job)

    assert calls.get("ocr") == 1, calls
    assert calls.get("translate") == 1, calls
    assert calls["ocr_kwargs"]["lang"] == "ch" and calls["ocr_kwargs"]["auto_crop"] is True

    row = db.tables["jobs"][0]
    assert row["status"] == "done", row["status"]
    # The OCR pass is 'transcribing' to the rest of the app: same lane, same
    # stall sweep, no enum migration.
    assert db.statuses[:3] == ["compiling", "transcribing", "translating"], db.statuses
    assert db.statuses[-1] == "done", db.statuses

    cues = db.tables["cues"]
    assert cues, "no cues inserted"
    assert all(c["transcript_text"] for c in cues), "cue missing transcript_text"
    assert all(c["source_text"].startswith("[vi] ") for c in cues), \
        "translation did not reach source_text"
    assert all(c["translated_at"] for c in cues), "translated_at not stamped"
    assert row["total_cues"] == len(cues) == row["done_cues"]

    src = [p for p in db.uploads if p.endswith("transcript.src.srt")]
    vi = [p for p in db.uploads if p.endswith("transcript.vi.srt")]
    assert src and vi, db.uploads
    assert db.uploads.index(src[0]) < db.uploads.index(vi[0]), \
        "the source subtitles must be safe before translation is attempted"
    assert row["transcript_src_path"] == src[0]
    assert row["srt_out_path"] == vi[0]

    qc = row["qc_summary"]
    assert qc["kind"] == "hardsub"
    assert qc["source_language"] == "Chinese" and qc["target_language"] == "Vietnamese"
    assert qc["cues_translated"] == len(cues) and qc["cues_untranslated"] == 0
    assert qc["scan_area_pct"] == [0.0, 70.0, 100.0, 19.0]
    assert qc["static_text_removed"] == ["Veo"]
    assert qc["glossary"] == {"曼云": "Mạn Vân"}, qc
    assert qc["translate_tokens"]["prompt_tokens"] == 100


@check("scene text and low-confidence noise never reach the cues table")
def _():
    db = make_db()
    job = new_job(db)
    run_job(db, job)

    texts = [c["transcript_text"] for c in db.tables["cues"]]
    assert "GV-6606" not in texts, "a licence plate became a cue"
    # 别怕 at 0.71 is under the 0.9 default and had no junk removed: dropped,
    # the same call the CLI makes.
    assert "别怕" not in texts, texts
    assert texts == ["你怎么在这里", "曼云 我们走"], texts
    idxs = [c["idx"] for c in db.tables["cues"]]
    assert idxs == [1, 2], f"cues must be renumbered after cleaning: {idxs}"

    row = db.tables["jobs"][0]
    assert row["qc_summary"]["cues_read"] == 4 and row["qc_summary"]["cues_dropped"] == 2
    # Every decision is on the job log with the reason, so the filter can be
    # loosened when it took a real line.
    ev = next(e for e in db.tables["job_events"] if e["message"].startswith("Dropped 2"))
    dropped = {d["text"]: d["why"] for d in ev["details"]["dropped"]}
    assert "GV-6606" in dropped and "CJK" in dropped["GV-6606"], dropped
    assert "别怕" in dropped and "confidence" in dropped["别怕"], dropped


@check("with the filter off, doubtful lines are kept and flagged for review")
def _():
    db = make_db()
    job = new_job(db, settings={"language": "Vietnamese", "ocr_lang": "ch",
                                "min_confidence": 0})
    run_job(db, job)

    cues = db.tables["cues"]
    texts = [c["transcript_text"] for c in cues]
    assert "别怕" in texts, texts
    # The no-CJK rule is about the script, not the confidence: the plate
    # still goes, whatever the threshold.
    assert "GV-6606" not in texts, texts
    flagged = [c for c in cues if c["status"] == "review"]
    assert [c["transcript_text"] for c in flagged] == ["别怕"], flagged
    assert all(c["note"] for c in flagged), flagged
    row = db.tables["jobs"][0]
    assert row["review_cues"] == 1 and row["qc_summary"]["cues_needing_review"] == 1


@check("a requeued job never reads the video twice")
def _():
    db = make_db()
    job = new_job(db)
    run_job(db, job)
    n_cues = len(db.tables["cues"])

    row = db.tables["jobs"][0]
    row["status"] = "queued"
    db.statuses.clear()
    db.uploads.clear()
    target = db.tables["cues"][0]
    target["translated_at"] = None
    target["source_text"] = target["transcript_text"]

    calls = run_job(db, row)
    assert calls.get("ocr") is None, "re-read a video whose lines were already committed"
    assert calls.get("translate") == 1, calls
    assert len(db.tables["cues"]) == n_cues, "cues were duplicated on requeue"
    assert "transcribing" not in db.statuses, db.statuses
    assert db.tables["jobs"][0]["status"] == "done"
    # The OCR facts from the first pass survive the second.
    qc = db.tables["jobs"][0]["qc_summary"]
    assert qc["cues_read"] == 4 and qc["scan_area_pct"] == [0.0, 70.0, 100.0, 19.0], qc
    events = [e["message"] for e in db.tables["job_events"]]
    assert any("1 line(s) were already translated" not in m and "already translated" in m
               for m in events) or any("Reusing" in m for m in events), events


@check("the glossary on the job is reused rather than rebuilt")
def _():
    db = make_db()
    job = new_job(db, settings={"language": "Vietnamese", "ocr_lang": "ch",
                                "glossary": {"曼云": "Mạn Vân", "凤凰": "Phượng Hoàng"}})
    calls = run_job(db, job)
    assert calls.get("glossary") is None, "rebuilt a glossary the job already carried"
    assert calls["translate_glossary"] == {"曼云": "Mạn Vân", "凤凰": "Phượng Hoàng"}


@check("without DEEPSEEK_API_KEY the subtitles are saved and the job says how to resume")
def _():
    db = make_db()
    job = new_job(db)
    try:
        run_job(db, job, api_key=None)
    except RuntimeError as e:
        assert "DEEPSEEK_API_KEY" in str(e) and "requeue" in str(e), e
    else:
        raise AssertionError("a missing key must surface")
    assert db.tables["cues"], "cues must be committed before the key is needed"
    assert any(p.endswith("transcript.src.srt") for p in db.uploads), db.uploads
    assert db.tables["jobs"][0]["transcript_src_path"]
    assert db.statuses[-1] == "translating", db.statuses


@check("a translation crash of any exception type reports the recovery fact")
def _():
    import json as _json

    db = make_db()
    job = new_job(db)

    def bad_json(cues, flush):
        raise _json.JSONDecodeError("Expecting value", "doc", 0)

    try:
        run_job(db, job, translate=bad_json)
    except RuntimeError as e:
        assert "subtitles are saved" in str(e) and "Expecting value" in str(e), e
    else:
        raise AssertionError("a translation crash must surface")
    assert any(p.endswith("transcript.src.srt") for p in db.uploads), db.uploads


@check("untranslated lines are reported and keep the text as read")
def _():
    db = make_db()
    job = new_job(db)

    def half(cues, flush):
        cues = list(cues)
        out = {cues[0].index: "[vi] only one"}
        if flush:
            flush(dict(out))
        return out, [c.index for c in cues[1:]]

    run_job(db, job, translate=half)
    row = db.tables["jobs"][0]
    qc = row["qc_summary"]
    assert qc["cues_translated"] == 1, qc
    assert qc["cues_untranslated"] == len(db.tables["cues"]) - 1, qc
    assert row["status"] == "done", "a partial translation still delivers a file"
    untouched = [c for c in db.tables["cues"] if not c["translated_at"]]
    assert untouched and all(c["source_text"] == c["transcript_text"] for c in untouched)
    events = [e["message"] for e in db.tables["job_events"]]
    assert any("could not be translated" in m for m in events), events
    assert any("came back missing" in m for m in events), events


@check("translation switched off delivers the subtitles as read, without the API")
def _():
    db = make_db()
    job = new_job(db, settings={"language": "Vietnamese", "ocr_lang": "ch", "translate": False})
    calls = run_job(db, job, api_key=None)
    assert calls.get("translate") is None and calls.get("glossary") is None, calls
    row = db.tables["jobs"][0]
    assert row["status"] == "done"
    assert "translating" not in db.statuses, db.statuses
    assert row["srt_out_path"].endswith("transcript.zh.srt"), row["srt_out_path"]
    qc = row["qc_summary"]
    assert qc["target_language"] == "Chinese" and qc["cues_untranslated"] == 0, qc
    assert row["done_cues"] == row["total_cues"] == len(db.tables["cues"])


@check("an interrupted read is discarded rather than extended")
def _():
    db = make_db()
    job = new_job(db, total_cues=9)
    for i in (1, 2):
        db.tables["cues"].append({
            "id": f"c{i}", "owner_id": "owner-1", "job_id": "job-1", "idx": i,
            "start_ms": i * 1000, "end_ms": i * 1000 + 800,
            "source_text": "partial", "transcript_text": "partial",
            "translated_at": None, "status": "pending"})
    calls = run_job(db, job)
    assert calls.get("ocr") == 1, "a partial read must be redone"
    assert ("cues", "delete", 2) in db.log, db.log
    assert not any(c["source_text"] == "partial" for c in db.tables["cues"])
    idxs = [c["idx"] for c in db.tables["cues"]]
    assert idxs == list(range(1, len(idxs) + 1)), idxs


@check("a pinned scan band and the OCR options reach hardsub_ocr")
def _():
    db = make_db()
    job = new_job(db, settings={"language": "Vietnamese", "ocr_lang": "en",
                                "crop": "0,70,100,19", "brightness_min": 220,
                                "ignore_text": ["Veo"], "keep_side_text": True})
    calls = run_job(db, job, lines=[
        {"index": 1, "start_ms": 100.0, "end_ms": 900.0, "text": "Hello there", "confidence": 0.95}])
    kw = calls["ocr_kwargs"]
    assert kw["lang"] == "en" and kw["crop"] == (0.0, 70.0, 100.0, 19.0) and kw["auto_crop"] is False
    assert kw["brightness_min"] == 220 and kw["ignore_text"] == ["Veo"] and kw["side_text"] is True
    # An English source has no Han characters; the no-CJK rule must not fire.
    assert [c["transcript_text"] for c in db.tables["cues"]] == ["Hello there"]
    assert db.tables["jobs"][0]["qc_summary"]["source_language"] == "English"


@check("a job with no video fails before touching the OCR")
def _():
    db = make_db()
    job = new_job(db, video_path=None)
    try:
        calls = run_job(db, job)
    except RuntimeError as e:
        assert "video" in str(e).lower(), e
    else:
        raise AssertionError(f"should have raised, ran {calls}")


@check("an unknown subtitle language is refused by name, not with SystemExit")
def _():
    db = make_db()
    job = new_job(db, settings={"language": "Vietnamese", "ocr_lang": "klingon"})
    try:
        run_job(db, job)
    except RuntimeError as e:
        assert "klingon" in str(e), e
    else:
        raise AssertionError("should have raised")


@check("a video with nothing readable fails with advice, after the read")
def _():
    db = make_db()
    job = new_job(db)
    try:
        run_job(db, job, lines=[])
    except RuntimeError as e:
        assert "scan band" in str(e), e
    else:
        raise AssertionError("should have raised")
    assert not db.tables["cues"]


@check("the uploaded .srt parses and carries the translation with the OCR timing")
def _():
    import tempfile
    from pathlib import Path

    db = make_db()
    job = new_job(db)
    run_job(db, job)
    blob = db.files["outputs"][db.tables["jobs"][0]["srt_out_path"]]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "out.srt"
        p.write_bytes(blob)
        cues = D.parse_srt(p)
    assert [c.text for c in cues] == ["[vi] 你怎么在这里", "[vi] 曼云 我们走"], cues
    assert cues[0].start == "00:00:01,000" and cues[0].end == "00:00:02,200", cues[0]
    assert [c.index for c in cues] == [1, 2]


def run():
    failed = 0
    for name, fn in checks:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}\n          {e}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"  ERROR {name}\n          {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(checks) - failed}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run())
