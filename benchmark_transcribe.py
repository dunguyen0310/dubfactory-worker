"""Measure the transcribe pipeline against burned-in subtitles, on your footage.

Every finished hardsub job is an answer key: frame-accurate subtitles, timed by
the show's own subtitlers, for a video that is still in storage. This script
scores transcripts of those same videos with eval_subtitles.py, so "does the
transcript match the video?" is answered with numbers instead of by eye.

    python benchmark_transcribe.py --baseline          # existing transcribe jobs
    python benchmark_transcribe.py --episodes 5        # the current pipeline
    python benchmark_transcribe.py --episodes 5 --vocals both   # A/B Demucs

--baseline pairs finished transcribe jobs with finished hardsub jobs of the
same title and scores what the transcribe job delivered — no GPU needed.
Without it, the pipeline in this checkout runs on the newest hardsub videos
(WhisperX and, realistically, a GPU needed) and is scored the same way.

Credentials are the worker's (SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY,
WORKER_EMAIL, WORKER_PASSWORD, or SUPABASE_SERVICE_ROLE_KEY). It only reads:
no job, cue or file in the app is created or changed.
"""

import argparse
import json
import statistics
import tempfile
from pathlib import Path

import eval_subtitles as E
import srt_dub as S

KEYS = ("appear", "disappear")


def _jobs(sb, kind):
    rows = (sb.table("jobs")
            .select("id,title,owner_id,video_path,transcript_src_path,created_at")
            .eq("kind", kind).eq("status", "done")
            .order("created_at", desc=True).limit(1000).execute().data) or []
    return [r for r in rows if r.get("transcript_src_path")]


def _download(sb, bucket, path, dest: Path) -> Path:
    dest.write_bytes(sb.storage.from_(bucket).download(path))
    return dest


def _row(title, result, cues_file=None):
    a, d = result["appear"] or {}, result["disappear"] or {}
    return {
        "title": title,
        "lines": f"{result['lines_found']}/{result['reference_lines']}",
        "appear_med": a.get("median_abs_s"), "appear_p90": a.get("p90_abs_s"),
        "appear_offset": a.get("signed_median_s"),
        "appear_0.5s": a.get("within_500ms"),
        "disappear_med": d.get("median_abs_s"),
        "same_cuts": result["boundaries_shared"], "cer": result["cer"],
        "cue_med_s": result["shape_hyp"].get("median_seconds"),
        "cue_max_chars": result["shape_hyp"].get("max_chars"),
        "file": cues_file,
    }


def _print(rows, label):
    print(f"\n== {label}")
    cols = ("appear_med", "appear_p90", "appear_offset", "appear_0.5s",
            "disappear_med", "same_cuts", "cer", "cue_med_s", "cue_max_chars")
    print(f"{'episode':34} {'lines':>7} " + " ".join(f"{c:>13}" for c in cols))
    for r in rows:
        print(f"{r['title'][:34]:34} {r['lines']:>7} "
              + " ".join(f"{'' if r[c] is None else r[c]:>13}" for c in cols))
    if rows:
        med = {c: statistics.median([r[c] for r in rows if r[c] is not None] or [0])
               for c in cols}
        print(f"{'MEDIAN over ' + str(len(rows)) + ' episode(s)':34} {'':>7} "
              + " ".join(f"{round(med[c], 3):>13}" for c in cols))


def baseline(sb, limit, work: Path, simplify):
    answers = {j["title"]: j for j in _jobs(sb, "hardsub")}
    pairs = [(t, answers[t["title"]]) for t in _jobs(sb, "transcribe")
             if t["title"] in answers][:limit]
    if not pairs:
        print("no finished transcribe job shares a title with a finished hardsub job")
        return []
    rows = []
    for tr, hs in pairs:
        ref = _download(sb, "outputs", hs["transcript_src_path"], work / f"{hs['id']}.ocr.srt")
        hyp = _download(sb, "outputs", tr["transcript_src_path"], work / f"{tr['id']}.asr.srt")
        res = E.evaluate(S.parse_srt(str(ref)), S.parse_srt(str(hyp)), simplify)
        rows.append(_row(tr["title"], res, hyp.name))
    _print(rows, "existing transcribe jobs vs the burned-in subtitles")
    return rows


def current(sb, episodes, vocals, model, work: Path, simplify):
    import transcribe_video as T
    modes = {"off": [False], "on": [True], "both": [False, True]}[vocals]
    jobs = [j for j in _jobs(sb, "hardsub") if j.get("video_path")][:episodes]
    rows = {m: [] for m in modes}
    for j in jobs:
        video = _download(sb, "videos", j["video_path"],
                          work / f"{j['id']}{Path(j['video_path']).suffix or '.mp4'}")
        ref = _download(sb, "outputs", j["transcript_src_path"], work / f"{j['id']}.ocr.srt")
        for isolate in modes:
            tag = "vocals" if isolate else "mix"
            print(f"\n-- {j['title']} ({tag})", flush=True)
            result = T.transcribe(str(video), model_name=model, prompt="auto",
                                  isolate=isolate, workdir=work / f"{j['id']}_{tag}",
                                  log=lambda m: print("   ", m, flush=True))
            cues = T.shape_cues(result["segments"], language=result["language"])
            hyp = work / f"{j['id']}.{tag}.src.srt"
            T.write_srt(cues, hyp)
            (work / f"{j['id']}.{tag}.words.json").write_text(
                json.dumps(T.words_json(result), ensure_ascii=False), encoding="utf-8")
            res = E.evaluate(S.parse_srt(str(ref)), S.parse_srt(str(hyp)), simplify)
            rows[isolate].append(_row(j["title"], res, hyp.name))
    for isolate, r in rows.items():
        _print(r, "this pipeline, " + ("vocals isolated" if isolate else "full mix")
               + " — vs the burned-in subtitles")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", action="store_true",
                   help="score existing transcribe jobs (no GPU needed)")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--vocals", choices=["off", "on", "both"], default="off",
                   help="transcribe the full mix, Demucs' vocals stem, or both")
    p.add_argument("--model", default="large-v3")
    p.add_argument("--workdir", default=None, help="where downloads and results go")
    args = p.parse_args()

    import worker as W                  # the worker's sign-in, nothing else
    sb = W.get_client()
    work = Path(args.workdir or tempfile.mkdtemp(prefix="bench_"))
    work.mkdir(parents=True, exist_ok=True)
    simplify = E._simplifier()
    if args.baseline:
        rows = baseline(sb, args.episodes, work, simplify)
    else:
        rows = current(sb, args.episodes, args.vocals, args.model, work, simplify)
    (work / "benchmark.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    print(f"\nfiles and benchmark.json in {work}")


if __name__ == "__main__":
    main()
