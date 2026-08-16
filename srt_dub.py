"""SRT-driven dubbing with OmniVoice — coverage-first.

Priorities, in order:
  1. Every word in the .srt is actually spoken (verified by ASR, retried until so)
  2. Every cue sounds like the same reference speaker (verified by timbre match)
  3. Timing follows the .srt where it can, and slips later where it cannot

Why it is built this way: forcing a clip shorter than its words need makes the
model DROP WORDS rather than speak faster. Measured on Vietnamese ad copy,
hard-forcing slot durations gave 65% word coverage; leaving duration free gave
97%. So duration is left free by default and the timeline absorbs the overrun.

    python srt_dub.py --srt movie.srt --ref-audio ref.wav --ref-text "..." \
        --language Vietnamese --output dub.wav

Add --emit-srt to get a corrected .srt whose timings match the audio produced.
"""

import argparse
import difflib
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 24000

GAP = 0.06               # minimum silence between pushed cues
OVERSHOOT = 1.12         # model renders ~12% longer than a requested duration
DRAWL_CAP = 1.4          # cap on stretching a short line to fill a long slot
WORD_FUZZ = 0.82         # difflib ratio above which two words count as the same

SYLLABIC_RE = re.compile(
    r"[ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
    re.IGNORECASE,
)
SYLLABLES_PER_SEC = 6.0
TIME_RE = re.compile(
    r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{1,3})"
)
TAG_RE = re.compile(r"<[^>]+>|\{[^}]+\}")
DIGIT_RE = re.compile(r"\d")


@dataclass
class Cue:
    index: int
    start: float
    end: float
    text: str
    # The cue exactly as written, markup included. `text` has tags stripped
    # because that is what gets spoken, but the tags carry meaning of their own
    # — subtitlers use <i> for narration and phone audio — and speakers.py needs
    # them. Defaulted so the four-argument constructor keeps working.
    raw: str = ""

    @property
    def slot(self) -> float:
        return self.end - self.start


# ---------------------------------------------------------------- srt parsing

def parse_srt(path: str) -> list[Cue]:
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    cues = []
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        m = text_start = None
        for i, ln in enumerate(lines[:2]):
            m = TIME_RE.search(ln)
            if m:
                text_start = i + 1
                break
        if not m:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(g) for g in m.groups())
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
        end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
        raw = " ".join(lines[text_start:]).strip()
        text = TAG_RE.sub("", raw).strip()
        if text and end > start:
            cues.append(Cue(len(cues) + 1, start, end, text, raw))
    cues.sort(key=lambda c: c.start)
    return cues


def fmt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def syllables(text: str) -> int:
    return len([t for t in re.split(r"\s+", text) if any(c.isalnum() for c in t)])


def natural_estimate(text: str) -> float:
    if SYLLABIC_RE.search(text):
        return max(0.6, 0.25 + syllables(text) / SYLLABLES_PER_SEC)
    return max(0.8, 0.35 + len(text) / 15.5)


# ------------------------------------------------------------- verification

# Whisper writes spoken Vietnamese numerals back as digits, so "tám" in the
# script never matches "8" in the transcript unless we expand them first.
VN_DIGITS = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín",
}
def _strip_tones(w: str) -> str:
    """Fold Vietnamese diacritics away: NFD splits tone/vowel marks into
    combining characters we can drop. 'đ' carries a stroke rather than a
    combining mark, so it needs handling of its own."""
    w = w.replace("đ", "d").replace("Đ", "D")
    return "".join(
        c for c in unicodedata.normalize("NFD", w)
        if unicodedata.category(c) != "Mn"
    )


def words(s: str) -> list[str]:
    out = []
    for w in re.sub(r"[^\w\s]", " ", s.lower()).split():
        if w.isdigit() and len(w) <= 2:
            out += [VN_DIGITS[d] for d in w if d in VN_DIGITS]
        elif w:
            out.append(w)
    return out


def coverage(src: str, heard: str) -> tuple[float, list[str], list[str]]:
    """Compare script words against what ASR heard.

    Returns (ratio, truly_missing, tone_variants). A word counts as spoken if it
    matches exactly, fuzzily, or once tone marks are stripped — Whisper routinely
    mis-renders Vietnamese diacritics (gập/gặp) on audio that is actually correct,
    and retrying those forever would be chasing a transcription artifact, not a
    synthesis fault. Tone variants are reported separately so real tone errors
    stay visible for a human to check.
    """
    want, pool = words(src), words(heard)
    missing, tonal = [], []
    for w in want:
        best, best_i = 0.0, -1
        for i, g in enumerate(pool):
            r = 1.0 if w == g else difflib.SequenceMatcher(None, w, g).ratio()
            if r > best:
                best, best_i = r, i
        if best >= WORD_FUZZ and best_i >= 0:
            pool.pop(best_i)
            continue
        # second chance: same word ignoring tone/diacritic marks
        sw = _strip_tones(w)
        hit = next((i for i, g in enumerate(pool) if _strip_tones(g) == sw), -1)
        if hit >= 0:
            tonal.append(w)
            pool.pop(hit)
        else:
            missing.append(w)
    spoken = len(want) - len(missing)
    return spoken / max(len(want), 1), missing, tonal


def timbre_vector(wav: np.ndarray) -> np.ndarray:
    """Cheap timbre fingerprint (mean MFCC) for speaker-consistency checks."""
    import librosa
    if wav.size < SR // 4:
        return np.zeros(20, dtype=np.float32)
    m = librosa.feature.mfcc(y=wav.astype(np.float32), sr=SR, n_mfcc=20)
    return m.mean(axis=1)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# -------------------------------------------------------------- generation

def generate_clips(cues, args):
    import torch
    from omnivoice import OmniVoice
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    print(f"loading model on {args.device} ...", flush=True)
    model = OmniVoice.from_pretrained(
        args.model, device_map=args.device, dtype=torch.float16
    )
    if not args.ref_text:
        print("no --ref-text; auto-transcribing reference (a hand-typed "
              "transcript clones more accurately) ...", flush=True)
    prompt = model.create_voice_clone_prompt(args.ref_audio, ref_text=args.ref_text)
    print(f"reference transcript: {prompt.ref_text!r}")

    ref_wav, _ = sf.read(args.ref_audio, dtype="float32", always_2d=False)
    if ref_wav.ndim > 1:
        ref_wav = ref_wav.mean(axis=1)
    ref_timbre = timbre_vector(ref_wav)

    kw = {}
    if args.language:
        kw["language"] = args.language
    if args.num_step != 32 or args.guidance_scale != 2.0:
        kw["generation_config"] = OmniVoiceGenerationConfig(
            num_step=args.num_step, guidance_scale=args.guidance_scale
        )

    model.load_asr_model()

    clips, scores = {}, {}
    pending = list(cues)
    for attempt in range(1, args.max_attempts + 1):
        # attempt 1 leaves duration free (highest coverage). Later attempts hand
        # the model progressively more room for the lines it truncated.
        if attempt == 1:
            durs = None
        else:
            durs = [natural_estimate(c.text) * (1.0 + 0.1 * (attempt - 1)) / args.overshoot
                    for c in pending]

        gen_kw = dict(kw)
        if durs is not None:
            gen_kw["duration"] = durs
        print(f"\nattempt {attempt}: generating {len(pending)} cue(s)"
              f"{' (free duration)' if durs is None else ' (extra room)'}", flush=True)

        audios = []
        for i in range(0, len(pending), max(1, args.batch_size)):
            chunk = pending[i:i + max(1, args.batch_size)]
            sub = dict(gen_kw)
            if durs is not None:
                sub["duration"] = durs[i:i + len(chunk)]
            audios += model.generate(
                text=[c.text for c in chunk],
                voice_clone_prompt=[prompt] * len(chunk),
                **sub,
            )

        still = []
        for cue, a in zip(pending, audios):
            wav = np.asarray(a, dtype=np.float32).squeeze()
            heard = model.transcribe((torch.from_numpy(wav), SR))
            if isinstance(heard, (list, tuple)):
                heard = heard[0]
            cov, missing, tonal = coverage(cue.text, str(heard))
            sim = cosine(ref_timbre, timbre_vector(wav))
            prev = scores.get(cue.index)
            # keep the best attempt so far, judged on coverage then timbre
            if prev is None or (cov, sim) > (prev[0], prev[1]):
                clips[cue.index] = wav
                scores[cue.index] = (cov, sim, str(heard), missing, tonal)
            if cov < args.min_coverage or sim < args.min_similarity:
                still.append(cue)
                why = []
                if cov < args.min_coverage:
                    why.append(f"missing {missing}")
                if sim < args.min_similarity:
                    why.append(f"timbre {sim:.2f}")
                print(f"  cue {cue.index}: {cov*100:.0f}% — {'; '.join(why)}")
        pending = still
        if not pending:
            print("  all cues pass")
            break
    if pending:
        print(f"\n{len(pending)} cue(s) still short after {args.max_attempts} "
              f"attempts; keeping their best take.")

    cov_all = np.mean([s[0] for s in scores.values()]) if scores else 0.0
    sim_all = np.mean([s[1] for s in scores.values()]) if scores else 0.0
    print(f"\nfinal word coverage {cov_all*100:.1f}%   mean timbre match {sim_all:.3f}")
    for i, (cov, sim, heard, missing, tonal) in sorted(scores.items()):
        if cov < 1.0 or sim < args.min_similarity:
            print(f"  cue {i}: cov {cov*100:.0f}% sim {sim:.2f} missing={missing} "
                  f"heard={heard[:50]!r}")
    tone_flags = {i: s[4] for i, s in scores.items() if s[4]}
    if tone_flags:
        print("\n  tone/diacritic variance (word was spoken; check these by ear —"
              " Whisper often mis-renders Vietnamese tones on correct audio):")
        for i, tv in sorted(tone_flags.items()):
            print(f"    cue {i}: {tv}")
    return clips, scores


# ---------------------------------------------------------------- assembly

def assemble(cues, clips, args):
    """Place clips sequentially: a cue starts at its .srt time, or right after
    the previous clip if that has run past it. Nothing is ever cut off."""
    placed, cursor = [], 0.0
    for cue in cues:
        clip = clips.get(cue.index)
        if clip is None or clip.size == 0:
            placed.append((cue, None, 0.0))
            continue
        start = max(cue.start, cursor)
        cursor = start + clip.size / SR + args.gap
        placed.append((cue, clip, start))

    end = max((s + c.size / SR for _, c, s in placed if c is not None), default=0.0)
    end = max(end, args.total_duration or 0.0)
    timeline = np.zeros(int(round(end * SR)) + SR, dtype=np.float32)
    for _, clip, start in placed:
        if clip is None:
            continue
        pos = int(round(start * SR))
        seg = timeline[pos:pos + clip.size]
        seg += clip[:seg.size]

    peak = float(np.max(np.abs(timeline))) if timeline.size else 0.0
    if peak > 1.0:
        timeline /= peak

    srt_end = max(c.end for c in cues)
    print("\ncue   srt start   placed     dur   drift")
    for cue, clip, start in placed:
        d = 0.0 if clip is None else clip.size / SR
        print(f"{cue.index:>3}  {cue.start:8.2f}s {start:8.2f}s {d:6.2f}s "
              f"{start - cue.start:+6.2f}s")
    print(f"\noutput {end:.2f}s vs .srt {srt_end:.2f}s "
          f"(+{max(0.0, end - srt_end):.2f}s)")

    digits = [c.index for c in cues if DIGIT_RE.search(c.text)]
    if digits:
        print(f"NOTE: cue(s) {digits} contain digits; spell numbers as words "
              "(e.g. 'thứ 8' -> 'thứ tám') — they are otherwise misread.")
    return timeline, placed


def write_srt(placed, path):
    with open(path, "w", encoding="utf-8") as f:
        for n, (cue, clip, start) in enumerate(placed, 1):
            if clip is None:
                continue
            f.write(f"{n}\n{fmt_ts(start)} --> "
                    f"{fmt_ts(start + clip.size / SR)}\n{cue.text}\n\n")
    print(f"wrote {path} (timings matched to the generated audio)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--srt", required=True)
    p.add_argument("--output", default="dub.wav")
    p.add_argument("--model", default="k2-fsa/OmniVoice")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ref-audio")
    p.add_argument("--ref-text")
    p.add_argument("--language", default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--min-coverage", type=float, default=1.0,
                   help="retry a cue until this fraction of its words is spoken")
    p.add_argument("--min-similarity", type=float, default=0.90,
                   help="retry a cue whose timbre match to the reference is below this")
    p.add_argument("--max-attempts", type=int, default=4)
    p.add_argument("--num-step", type=int, default=32)
    p.add_argument("--guidance-scale", type=float, default=2.0)
    p.add_argument("--overshoot", type=float, default=OVERSHOOT)
    p.add_argument("--gap", type=float, default=GAP)
    p.add_argument("--total-duration", type=float, default=None)
    p.add_argument("--emit-srt", help="write a .srt matching the generated timings")
    p.add_argument("--clips-dir")
    args = p.parse_args()

    cues = parse_srt(args.srt)
    if not cues:
        sys.exit(f"no cues parsed from {args.srt}")
    print(f"parsed {len(cues)} cues, span {cues[0].start:.1f}s -> {cues[-1].end:.1f}s")

    if args.clips_dir:
        d = Path(args.clips_dir)
        clips = {}
        for c in cues:
            for cand in (d / f"{c.index}.wav", d / f"{c.index:04d}.wav"):
                if cand.exists():
                    w, sr = sf.read(cand, dtype="float32", always_2d=False)
                    if w.ndim > 1:
                        w = w.mean(axis=1)
                    clips[c.index] = w
                    break
    else:
        if not args.ref_audio:
            sys.exit("need --ref-audio (or --clips-dir)")
        clips, _ = generate_clips(cues, args)

    timeline, placed = assemble(cues, clips, args)
    sf.write(args.output, timeline, SR)
    print(f"wrote {args.output} ({timeline.size / SR:.2f}s)")
    if args.emit_srt:
        write_srt(placed, args.emit_srt)


if __name__ == "__main__":
    main()
