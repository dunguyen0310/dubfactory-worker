"""Score a subtitle file against a reference: timing, shape and text.

The best reference there is for this pipeline is free: a video with burned-in
subtitles, read back by hardsub_ocr.py. Those subtitles are frame-accurate and
were timed by the show's own subtitlers, so every hardsub job is ground truth
for the transcribe pipeline on the same episode:

    python eval_subtitles.py --ref ep1.ocr.srt --hyp ep1.src.srt
    python eval_subtitles.py --ref ref.srt --hyp hyp.srt --json
    python eval_subtitles.py --hyp ep1.vi.srt                # shape only

Both files must be in the same language for timing and text to be compared;
the text is matched character by character (punctuation, spaces and case
ignored; Traditional folded to Simplified when OpenCC is installed), and each
reference line is then followed to wherever the hypothesis put its words.

What it reports, in the order a viewer notices it:
  appear      how early (-) or late (+) each reference line's text comes on
              screen in the hypothesis: the start of the hypothesis cue that
              holds the line's first word, minus the reference cue's start.
              A hypothesis cue that swallows the previous sentence shows the
              line seconds early; that is exactly what this catches.
  disappear   the same for when the line's last word leaves the screen.
  boundaries  the share of reference cues the hypothesis also starts a cue
              with the same word: how alike the two are cut.
  shape       cue count, median duration and length, the longest cue, and the
              reading speed — for each file.
  text        character error rate of the hypothesis against the reference.
"""

import argparse
import json
import statistics
import sys
import unicodedata

import srt_dub as S


def _simplifier():
    try:
        from opencc import OpenCC
    except ImportError:
        return None
    for config in ("t2s", "t2s.json"):
        try:
            return OpenCC(config).convert
        except Exception:
            continue
    return None


def _norm(text: str, simplify=None) -> str:
    if simplify:
        text = simplify(text)
    return "".join(ch for ch in text.lower()
                   if unicodedata.category(ch)[0] not in "PSZC")


def _stream(cues, simplify):
    """All cues' characters as one string, with each character's cue."""
    chars, owner, spans = [], [], []
    for k, c in enumerate(cues):
        norm = _norm(c.text, simplify)
        spans.append((len(chars), len(chars) + len(norm)))
        chars.extend(norm)
        owner.extend([k] * len(norm))
    return "".join(chars), owner, spans


def _match(ref: str, hyp: str):
    """Matched character pairs (ref index -> hyp index) and the edit distance."""
    try:
        from rapidfuzz.distance import Levenshtein
    except ImportError:
        import difflib
        sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
        pairs = {b.a + k: b.b + k for b in sm.get_matching_blocks()
                 for k in range(b.size)}
        # Not Levenshtein, but the same scale: every unmatched character on
        # either side is one edit.
        return pairs, len(ref) + len(hyp) - 2 * len(pairs)
    pairs = {}
    for op in Levenshtein.opcodes(ref, hyp):
        if op.tag == "equal":
            for k in range(op.src_end - op.src_start):
                pairs[op.src_start + k] = op.dest_start + k
    return pairs, Levenshtein.distance(ref, hyp)


def _dist(values):
    """Median, 90th percentile and within-window shares of |error| (seconds)."""
    if not values:
        return None
    mags = sorted(abs(v) for v in values)
    p90 = mags[min(len(mags) - 1, int(round(0.9 * (len(mags) - 1))))]
    return {
        "n": len(values),
        "median_abs_s": round(statistics.median(mags), 3),
        "p90_abs_s": round(p90, 3),
        "signed_median_s": round(statistics.median(values), 3),
        "within_100ms": round(sum(m <= 0.1 for m in mags) / len(mags), 3),
        "within_250ms": round(sum(m <= 0.25 for m in mags) / len(mags), 3),
        "within_500ms": round(sum(m <= 0.5 for m in mags) / len(mags), 3),
    }


def shape(cues, cps_chars=None) -> dict:
    """How subtitle-shaped a cue list is. `cps_chars` counts characters for
    reading speed (default: every non-space character)."""
    if not cues:
        return {"cues": 0}
    secs = sorted(c.end - c.start for c in cues)
    lens = sorted(len(c.text.replace(" ", "")) for c in cues)
    cps = sorted((cps_chars or (lambda t: len(t.replace(" ", ""))))(c.text)
                 / max(c.end - c.start, 0.001) for c in cues)
    mid = len(cues) // 2
    return {
        "cues": len(cues),
        "median_seconds": round(secs[mid], 2),
        "max_seconds": round(secs[-1], 2),
        "median_chars": lens[mid],
        "max_chars": lens[-1],
        "median_cps": round(cps[mid], 1),
        "p90_cps": round(cps[min(len(cps) - 1, int(0.9 * (len(cps) - 1)))], 1),
        "seconds_covered": round(sum(secs), 1),
    }


def evaluate(ref_cues, hyp_cues, simplify=None) -> dict:
    """Compare two cue lists (srt_dub.Cue-like: .start .end .text)."""
    ref, rown, rspans = _stream(ref_cues, simplify)
    hyp, hown, hspans = _stream(hyp_cues, simplify)
    pairs, distance = _match(ref, hyp)

    appear, disappear, same_start = [], [], 0
    judged = 0
    for k, (a, b) in enumerate(rspans):
        hit = [i for i in range(a, b) if i in pairs]
        if not hit:
            continue
        judged += 1
        first, last = pairs[hit[0]], pairs[hit[-1]]
        h_first, h_last = hyp_cues[hown[first]], hyp_cues[hown[last]]
        appear.append(h_first.start - ref_cues[k].start)
        disappear.append(h_last.end - ref_cues[k].end)
        if hit[0] == a and hspans[hown[first]][0] == first:
            same_start += 1

    return {
        "reference_lines": len(ref_cues),
        "lines_found": judged,
        "appear": _dist(appear),
        "disappear": _dist(disappear),
        "boundaries_shared": round(same_start / judged, 3) if judged else None,
        "cer": round(distance / len(ref), 3) if ref else None,
        "shape_ref": shape(ref_cues),
        "shape_hyp": shape(hyp_cues),
    }


def _fmt(d):
    if not d:
        return "n/a"
    return (f"median {d['median_abs_s']:.2f}s, p90 {d['p90_abs_s']:.2f}s, "
            f"offset {d['signed_median_s']:+.2f}s, "
            f"within 0.25s {d['within_250ms'] * 100:.0f}%, "
            f"within 0.5s {d['within_500ms'] * 100:.0f}%")


def report(r: dict) -> str:
    lines = [
        f"reference lines found in the hypothesis: {r['lines_found']}"
        f"/{r['reference_lines']}",
        f"appear     {_fmt(r['appear'])}",
        f"disappear  {_fmt(r['disappear'])}",
        "same cue boundaries: "
        + (f"{r['boundaries_shared'] * 100:.0f}%" if r["boundaries_shared"]
           is not None else "n/a"),
        "text CER: " + (f"{r['cer'] * 100:.1f}%" if r["cer"] is not None
                         else "n/a"),
        f"shape ref: {r['shape_ref']}",
        f"shape hyp: {r['shape_hyp']}",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", help="reference .srt (e.g. hardsub OCR output)")
    p.add_argument("--hyp", required=True, help=".srt to score")
    p.add_argument("--json", action="store_true", help="print JSON")
    args = p.parse_args()

    hyp = S.parse_srt(args.hyp)
    if not args.ref:
        out = {"shape_hyp": shape(hyp)}
        print(json.dumps(out, ensure_ascii=False, indent=1) if args.json
              else f"shape: {out['shape_hyp']}")
        return
    ref = S.parse_srt(args.ref)
    if not ref:
        sys.exit(f"no cues in {args.ref}")
    result = evaluate(ref, hyp, _simplifier())
    print(json.dumps(result, ensure_ascii=False, indent=1) if args.json
          else report(result))


if __name__ == "__main__":
    main()
