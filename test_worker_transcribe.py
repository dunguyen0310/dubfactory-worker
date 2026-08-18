"""Orchestration tests for worker.transcribe_job, against a fake Supabase.

The GPU half cannot run here — no whisperx install, no CUDA, no episode — but
the half that goes wrong quietly can: which statuses are written and in what
order, whether the cues carry a transcript, whether both .srt files are
uploaded, and above all whether a requeued job re-runs the expensive ASR it
already paid for.

`transcribe_video.transcribe` and `translate_cues` are stubbed, so this needs no
GPU, no model download and no API key:

    python test_worker_transcribe.py
"""

import sys

import transcribe_video as T
import worker as W


# ------------------------------------------------------------- fake supabase

# Nullable columns the real tables have, so an inserted row looks like one that
# came back from Postgres rather than only the keys the worker happened to set.
COLUMN_DEFAULTS = {
    "cues": ("transcript_text", "translated_at", "final_text", "speaker",
             "voice_id", "audio_path", "rendered_ms", "cer", "note"),
    "jobs": ("transcript_src_path", "srt_out_path", "wav_path", "mp4_path",
             "qc_summary", "adapt_summary", "error", "heartbeat_at",
             "claimed_at", "review_cues", "done_cues"),
}


class Result:
    def __init__(self, data):
        self.data = data


class Query:
    """One chained supabase-py call. Records what it was asked to do."""

    def __init__(self, db, table, op, payload=None):
        self.db, self.table, self.op, self.payload = db, table, op, payload
        self.filters = {}

    def select(self, *a, **k):
        return self

    def eq(self, col, val):
        self.filters[col] = val
        return self

    def order(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def single(self):
        return self

    def _match(self, row):
        return all(row.get(k) == v for k, v in self.filters.items())

    def execute(self):
        rows = self.db.tables.setdefault(self.table, [])
        if self.op == "select":
            hit = [r for r in rows if self._match(r)]
            hit.sort(key=lambda r: r.get("idx", 0))
            return Result(hit)
        if self.op == "insert":
            new = self.payload if isinstance(self.payload, list) else [self.payload]
            for r in new:
                r = dict(r)
                r.setdefault("id", f"{self.table}-{len(rows) + 1}")
                # A real INSERT leaves unlisted columns present-and-NULL, not
                # absent. Mimicking that is what lets the tests below index
                # columns directly instead of .get()-ing them and accidentally
                # passing on a column the worker never wrote.
                for col in COLUMN_DEFAULTS.get(self.table, ()):
                    r.setdefault(col, None)
                rows.append(r)
            self.db.log.append((self.table, "insert", len(new)))
            return Result(new)
        if self.op == "upsert":
            rows.append(dict(self.payload))
            return Result([self.payload])
        if self.op == "update":
            hit = [r for r in rows if self._match(r)]
            for r in hit:
                r.update(self.payload)
            if self.table == "jobs" and "status" in self.payload:
                self.db.statuses.append(self.payload["status"])
            self.db.log.append((self.table, "update", tuple(self.payload)))
            return Result(hit)
        if self.op == "delete":
            hit = [r for r in rows if self._match(r)]
            for r in hit:
                rows.remove(r)
            self.db.log.append((self.table, "delete", len(hit)))
            return Result(hit)
        raise AssertionError(f"unhandled op {self.op}")


class Bucket:
    def __init__(self, db, name):
        self.db, self.name = db, name

    def download(self, path):
        if self.name not in self.db.files:
            raise RuntimeError(f"no bucket {self.name}")
        return self.db.files[self.name].get(path, b"\x00fake media\x00")

    def upload(self, path, data, opts=None):
        self.db.files.setdefault(self.name, {})[path] = data
        self.db.uploads.append(path)
        return Result([{"path": path}])


class Storage:
    def __init__(self, db):
        self.db = db

    def from_(self, name):
        return Bucket(self.db, name)


class FakeDB:
    """Every column exists and every table is writable — the migration is
    assumed run, because the point here is the orchestration, not degradation.

    `files` is the blob store; `storage` is the API surface over it, matching
    supabase-py where `sb.storage` is an attribute and not a dict.
    """

    def __init__(self):
        self.tables = {"jobs": [], "cues": [], "job_events": [],
                       "render_workers": [], "voices": []}
        self.files = {"videos": {}, "outputs": {}}
        self.storage = Storage(self)
        self.uploads = []
        self.statuses = []
        self.log = []

    def table(self, name):
        return _TableHandle(self, name)

    def rpc(self, *a, **k):
        return Query(self, "rpc", "select")


class _TableHandle:
    def __init__(self, db, name):
        self.db, self.name = db, name

    def select(self, *a, **k):
        return Query(self.db, self.name, "select")

    def insert(self, payload):
        return Query(self.db, self.name, "insert", payload)

    def upsert(self, payload):
        return Query(self.db, self.name, "upsert", payload)

    def update(self, payload):
        return Query(self.db, self.name, "update", payload)

    def delete(self):
        return Query(self.db, self.name, "delete")


def make_db():
    return FakeDB()


# -------------------------------------------------------------------- fixture

SEGMENTS = [
    {"start": 0.0, "end": 2.4, "text": "Hello and welcome to the show",
     "words": [{"word": w, "start": 0.4 * i, "end": 0.4 * i + 0.35,
                "score": 0.93}
               for i, w in enumerate("Hello and welcome to the show".split())]},
    {"start": 3.0, "end": 5.2, "text": "Today we talk about dubbing",
     "words": [{"word": w, "start": 3.0 + 0.4 * i, "end": 3.0 + 0.4 * i + 0.35,
                "score": 0.9}
               for i, w in enumerate("Today we talk about dubbing".split())]},
]


def install_stubs(monkey, *, translate=True, calls=None):
    """Replace the GPU and API halves with recorded fakes."""
    calls = calls if calls is not None else {}

    def fake_transcribe(media, **kw):
        calls["transcribe"] = calls.get("transcribe", 0) + 1
        return {"language": "en", "segments": SEGMENTS, "alignment": "word",
                "duration": 5.2}

    def fake_translate(cues, **kw):
        calls["translate"] = calls.get("translate", 0) + 1
        if not translate:
            raise T_NoCredentials("no API key configured")
        out = {T._cue_index(c): f"[vi] {T._cue_text(c)}" for c in cues}
        # The real translate_cues persists per batch through this callback,
        # and the worker now relies on it — a stub that only returns would
        # leave every cue unstamped and fail the assertions that matter.
        if kw.get("flush"):
            kw["flush"](out)
        return out

    monkey.append((T, "transcribe", T.transcribe))
    monkey.append((T, "translate_cues", T.translate_cues))
    T.transcribe = fake_transcribe
    T.translate_cues = fake_translate
    return calls


class T_NoCredentials(RuntimeError):
    pass


def new_job(db, **over):
    job = {"id": "job-1", "owner_id": "owner-1", "title": "Ep 1",
           "kind": "transcribe", "status": "queued",
           "video_path": "owner-1/ep1.mp4", "voice_id": None,
           "settings": {"language": "Vietnamese"}, "total_cues": 0,
           "qc_summary": None}
    job.update(over)
    db.tables["jobs"].append(job)
    return job


# ---------------------------------------------------------------------- tests

checks = []


def check(name):
    def wrap(fn):
        checks.append((name, fn))
        return fn
    return wrap


def run_job(db, job, **stub_kw):
    monkey, calls = [], {}
    install_stubs(monkey, calls=calls, **stub_kw)
    W._COLS.clear()
    W._beat.update(at=0.0, on=False)     # presence is not under test
    W._touch["at"] = 0.0
    try:
        W.transcribe_job(db, job)
    finally:
        for obj, attr, orig in monkey:
            setattr(obj, attr, orig)
    return calls


@check("a fresh job transcribes, translates, and uploads both .srt files")
def _():
    db = make_db()
    job = new_job(db)
    calls = run_job(db, job)

    assert calls.get("transcribe") == 1, calls
    assert calls.get("translate") == 1, calls

    row = db.tables["jobs"][0]
    assert row["status"] == "done", row["status"]
    # Stages must be walked in pipeline order, and the two new ones must appear.
    assert db.statuses[:3] == ["compiling", "transcribing", "translating"], db.statuses
    assert db.statuses[-1] == "done", db.statuses

    cues = db.tables["cues"]
    assert cues, "no cues inserted"
    assert all(c["transcript_text"] for c in cues), "cue missing transcript_text"
    assert all(c["source_text"].startswith("[vi] ") for c in cues), \
        "translation did not reach source_text"
    assert all(c["translated_at"] for c in cues), "translated_at not stamped"
    assert row["total_cues"] == len(cues), (row["total_cues"], len(cues))

    # Both deliverables, and the source one before the translated one.
    src = [p for p in db.uploads if p.endswith("transcript.src.srt")]
    vi = [p for p in db.uploads if p.endswith("transcript.vi.srt")]
    assert src and vi, db.uploads
    assert db.uploads.index(src[0]) < db.uploads.index(vi[0]), \
        "the source transcript must be safe before translation is attempted"
    assert row["transcript_src_path"] == src[0]
    assert row["srt_out_path"] == vi[0]

    qc = row["qc_summary"]
    assert qc["source_language"] == "en" and qc["target_language"] == "Vietnamese"
    assert qc["alignment"] == "word"
    assert qc["cues_translated"] == len(cues) and qc["cues_untranslated"] == 0


@check("the uploaded .srt is valid and carries the translation")
def _():
    import srt_dub as S
    import tempfile
    from pathlib import Path

    db = make_db()
    job = new_job(db)
    run_job(db, job)

    blob = db.files["outputs"][db.tables["jobs"][0]["srt_out_path"]]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "out.srt"
        p.write_bytes(blob)
        cues = S.parse_srt(str(p))
    assert cues, "the emitted .srt does not parse"
    assert all(c.text.startswith("[vi] ") for c in cues), cues[0].text
    for a, b in zip(cues, cues[1:]):
        assert b.start >= a.end, f"overlapping cues in output: {a} {b}"


@check("a requeued job never pays for ASR twice")
def _():
    db = make_db()
    job = new_job(db)
    run_job(db, job)
    n_cues = len(db.tables["cues"])

    # Simulate the re-roll/requeue path: the job goes back to the queue with its
    # transcript intact. Only the translation should run again — and only for
    # the cue whose translation was cleared.
    row = db.tables["jobs"][0]
    row["status"] = "queued"
    db.statuses.clear()
    db.uploads.clear()
    target = db.tables["cues"][0]
    target["translated_at"] = None
    target["source_text"] = target["transcript_text"]

    calls = run_job(db, row)
    assert calls.get("transcribe") is None, \
        "re-transcribed a job that already had its transcript"
    assert calls.get("translate") == 1, calls
    assert len(db.tables["cues"]) == n_cues, "cues were duplicated on requeue"
    assert "transcribing" not in db.statuses, db.statuses
    assert db.tables["jobs"][0]["status"] == "done"
    # The metadata from the first pass survives, rather than coming back None.
    assert db.tables["jobs"][0]["qc_summary"]["source_language"] == "en"


@check("an interrupted transcript is discarded rather than extended")
def _():
    db = make_db()
    # total_cues says 9 but only 2 rows landed: the insert died half way.
    job = new_job(db, total_cues=9)
    for i in (1, 2):
        db.tables["cues"].append({
            "id": f"c{i}", "owner_id": "owner-1", "job_id": "job-1", "idx": i,
            "start_ms": i * 1000, "end_ms": i * 1000 + 800,
            "source_text": "partial", "transcript_text": "partial",
            "translated_at": None, "status": "pending"})

    calls = run_job(db, job)
    assert calls.get("transcribe") == 1, "a partial transcript must be redone"
    assert ("cues", "delete", 2) in db.log, db.log
    idxs = sorted(c["idx"] for c in db.tables["cues"])
    assert idxs == list(range(1, len(idxs) + 1)), f"cue numbering broken: {idxs}"
    assert not any(c["source_text"] == "partial" for c in db.tables["cues"])


@check("a job with no media fails before touching the GPU")
def _():
    db = make_db()
    job = new_job(db, video_path=None)
    try:
        calls = run_job(db, job)
    except RuntimeError as e:
        assert "media" in str(e).lower(), e
    else:
        raise AssertionError(f"should have raised, ran {calls}")


@check("untranslated lines are reported and keep their transcript")
def _():
    db = make_db()
    job = new_job(db)

    monkey, calls = [], {}
    install_stubs(monkey, calls=calls)
    # A provider that translates only the first cue — the partial-failure case.
    def half(cues, **kw):
        cues = list(cues)
        out = {T._cue_index(cues[0]): "[vi] only one"}
        if kw.get("flush"):
            kw["flush"](out)
        return out
    T.translate_cues = half
    W._COLS.clear()
    W._beat.update(at=0.0, on=False)
    W._touch["at"] = 0.0
    try:
        W.transcribe_job(db, job)
    finally:
        for obj, attr, orig in monkey:
            setattr(obj, attr, orig)

    row = db.tables["jobs"][0]
    qc = row["qc_summary"]
    assert qc["cues_translated"] == 1, qc
    assert qc["cues_untranslated"] == len(db.tables["cues"]) - 1, qc
    assert row["status"] == "done", "a partial translation still delivers a file"
    # The untranslated lines must still be in the file, as transcribed.
    untouched = [c for c in db.tables["cues"] if not c["translated_at"]]
    assert untouched and all(
        c["source_text"] == c["transcript_text"] for c in untouched)
    events = [e["message"] for e in db.tables["job_events"]]
    assert any("could not be translated" in m for m in events), events


@check("low-confidence lines are flagged for review on the job")
def _():
    db = make_db()
    job = new_job(db)

    monkey, calls = [], {}
    install_stubs(monkey, calls=calls)
    def murky(media, **kw):
        segs = [dict(s) for s in SEGMENTS]
        segs[1] = dict(segs[1], words=[dict(w, score=0.3)
                                       for w in segs[1]["words"]])
        return {"language": "en", "segments": segs, "alignment": "word",
                "duration": 5.2}
    T.transcribe = murky
    W._COLS.clear()
    W._beat.update(at=0.0, on=False)
    W._touch["at"] = 0.0
    try:
        W.transcribe_job(db, job)
    finally:
        for obj, attr, orig in monkey:
            setattr(obj, attr, orig)

    flagged = [c for c in db.tables["cues"] if c["status"] == "review"]
    assert flagged, "a 0.3-score line should be flagged, not shipped silently"
    assert all(c["note"] for c in flagged), flagged
    assert db.tables["jobs"][0]["qc_summary"]["cues_needing_review"] == len(flagged)
    assert db.tables["jobs"][0]["review_cues"] == len(flagged)


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
