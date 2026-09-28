"""Tests for eval_subtitles.py: the scorer must report exactly what a viewer
would see — a subtitle that comes up early because it swallowed the line
before it, a constant offset, a wrong character.

    python test_eval_subtitles.py
"""

from collections import namedtuple

import eval_subtitles as E

Cue = namedtuple("Cue", "start end text")

REF = [Cue(1.0, 2.2, "你怎么在这里"), Cue(2.5, 3.1, "我来找你"),
       Cue(3.4, 5.0, "你不该来的")]

checks = []


def check(name):
    def wrap(fn):
        checks.append((name, fn))
        return fn
    return wrap


@check("identical files score perfectly")
def _():
    r = E.evaluate(REF, REF)
    assert r["cer"] == 0 and r["boundaries_shared"] == 1.0, r
    assert r["appear"]["median_abs_s"] == 0 and r["disappear"]["p90_abs_s"] == 0, r
    assert r["lines_found"] == 3


@check("a constant offset shows as the signed median")
def _():
    hyp = [Cue(c.start + 0.3, c.end + 0.3, c.text) for c in REF]
    r = E.evaluate(REF, hyp)
    assert r["appear"]["signed_median_s"] == 0.3, r["appear"]
    assert r["appear"]["within_250ms"] == 0 and r["appear"]["within_500ms"] == 1.0


@check("a cue that swallows three lines shows them early and late")
def _():
    hyp = [Cue(1.0, 5.0, "你怎么在这里我来找你你不该来的")]
    r = E.evaluate(REF, hyp)
    assert r["cer"] == 0, r
    # Line 2 appears 1.5 s before it is spoken, line 3 2.4 s before.
    assert r["appear"]["signed_median_s"] == -1.5, r["appear"]
    assert r["boundaries_shared"] == round(1 / 3, 3), r
    assert r["shape_hyp"]["max_seconds"] == 4.0


@check("wrong characters raise the error rate; punctuation and spacing do not")
def _():
    hyp = [Cue(1.0, 2.2, "你怎么在这儿？"), Cue(2.5, 3.1, "我 来找你。"),
           Cue(3.4, 5.0, "你不该来的！")]
    r = E.evaluate(REF, hyp)
    assert r["cer"] == round(1 / 15, 3), r["cer"]
    assert r["boundaries_shared"] == 1.0


@check("Traditional characters match Simplified when OpenCC is installed")
def _():
    simplify = E._simplifier()
    if simplify is None:
        print("          (skipped: OpenCC not installed)")
        return
    hyp = [Cue(1.0, 2.2, "你怎麼在這裡"), Cue(2.5, 3.1, "我來找你"),
           Cue(3.4, 5.0, "你不該來的")]
    assert E.evaluate(REF, hyp, simplify)["cer"] == 0


@check("an empty hypothesis reports nothing found rather than crashing")
def _():
    r = E.evaluate(REF, [])
    assert r["lines_found"] == 0 and r["appear"] is None and r["cer"] == 1.0, r
    assert E.shape([]) == {"cues": 0}


@check("the baseline benchmark pairs jobs by title and only reads")
def _():
    import tempfile
    from pathlib import Path
    import benchmark_transcribe as B
    from test_worker_transcribe import make_db

    srt = lambda cues: "".join(                       # noqa: E731
        f"{n}\n00:00:0{int(s)},{int(s % 1 * 1000):03d} --> 00:00:0{int(e)},"
        f"{int(e % 1 * 1000):03d}\n{t}\n\n" for n, (s, e, t) in enumerate(cues, 1)
    ).encode("utf-8")
    db = make_db()
    db.tables["jobs"] += [
        {"id": "hs1", "title": "Ep01", "kind": "hardsub", "status": "done",
         "owner_id": "o", "transcript_src_path": "o/hs1/transcript.src.srt"},
        {"id": "tr1", "title": "Ep01", "kind": "transcribe", "status": "done",
         "owner_id": "o", "transcript_src_path": "o/tr1/transcript.src.srt"},
        {"id": "tr2", "title": "Ep02", "kind": "transcribe", "status": "done",
         "owner_id": "o", "transcript_src_path": "o/tr2/transcript.src.srt"},
    ]
    db.files["outputs"].update({
        "o/hs1/transcript.src.srt": srt([(1.0, 2.2, "你怎么在这里"), (2.5, 3.1, "我来找你")]),
        "o/tr1/transcript.src.srt": srt([(1.0, 3.1, "你怎么在这里我来找你")]),
    })
    before = list(db.log)
    with tempfile.TemporaryDirectory() as d:
        rows = B.baseline(db, 10, Path(d), None)
    assert [r["title"] for r in rows] == ["Ep01"], rows      # Ep02 has no answer key
    assert rows[0]["appear_offset"] == -0.75, rows            # line 2 shown 1.5 s early
    assert rows[0]["cer"] == 0 and rows[0]["same_cuts"] == 0.5, rows
    assert db.log == before, "the benchmark must not write to the database"


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
            print(f"  ERROR {name}\n          {type(e).__name__}: {e}")
    print(f"\n{len(checks) - failed}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
