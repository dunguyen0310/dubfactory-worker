"""Voice library: QC, encode, audition and cache a cloned voice.

Staff upload a sample from the web UI (`voices` row, status='uploaded'); this
module turns it into a library voice that any worker can load instantly.

Pipeline per voice:
    1. QC      — duration, level, clipping, bandwidth, pitch consistency
    2. Transcribe — Whisper, unless staff typed a transcript (typed is better)
    3. Encode  — build the VoiceClonePrompt and cache it to storage as .pt
    4. Audition — synthesise a short line so staff can hear the clone
    5. Save    — voices.status = 'ready'

Caching step 3 is the point: encoding costs GPU seconds and the result is
identical every time, so it is wasteful to redo it in each Colab session.
"""

import io
import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

import srt_dub as S

SR = S.SR

# Sample-quality thresholds. These are advisory except where noted; the model
# still runs on a bad sample, it just clones worse.
MIN_SECONDS = 3.0
MAX_SECONDS = 20.0          # beyond this the model itself warns and slows down
IDEAL_RANGE = (5.0, 10.0)
MIN_SNR_DB = 15.0
MIN_BANDWIDTH_HZ = 6000.0   # below this the sample was heavily compressed
CLIP_PEAK = 0.99
PITCH_SHIFT_WARN = 45.0     # Hz between halves suggests two speakers

AUDITION_TEXT = {
    "Vietnamese": "Xin chào, đây là giọng đọc mẫu cho thư viện giọng nói.",
    "English": "Hello, this is a sample of how this voice will sound.",
}

# The seed line a designed voice is grown from. Longer than the audition line:
# it becomes the clone reference, and 3–10s of speech clones far better than a
# two-second fragment.
DESIGN_SEED_TEXT = {
    "Vietnamese": (
        "Xin chào các bạn, hôm nay chúng ta sẽ cùng nhau khám phá "
        "một sản phẩm rất thú vị và hoàn toàn mới."
    ),
    "English": (
        "Hello there, today we are going to take a look at something "
        "genuinely new and rather interesting."
    ),
}


def design_voice(model, instruct: str, language: str, workdir: Path) -> dict:
    """Build a synthetic voice from an attribute description, then FREEZE it.

    `instruct` alone re-rolls a different speaker on every call, so it cannot
    back a library voice on its own. Generating one seed clip and turning that
    into a VoiceClonePrompt pins the identity: every later job reuses the same
    narrator. The seed clip doubles as the stored 'sample'.
    """
    seed_text = DESIGN_SEED_TEXT.get(language, DESIGN_SEED_TEXT["English"])
    kw = {"language": language} if language else {}

    audio = model.generate(text=[seed_text], instruct=instruct, **kw)
    seed = np.asarray(audio[0], dtype=np.float32).squeeze()
    seed_file = workdir / "designed_seed.wav"
    sf.write(seed_file, seed, SR)

    # Feed the exact text we asked for rather than re-transcribing: it is
    # already known, and a correct transcript conditions the clone better.
    import torch

    prompt = model.create_voice_clone_prompt((torch.from_numpy(seed), SR),
                                             ref_text=seed_text)
    return {"seed_file": seed_file, "seed_text": seed_text, "prompt": prompt}


def analyse_sample(path: str) -> dict:
    """Measure a reference clip and return QC evidence + human-readable warnings."""
    import librosa

    y, _ = librosa.load(path, sr=SR, mono=True)
    dur = len(y) / SR
    warnings: list[str] = []
    if y.size == 0:
        return {"duration_s": 0.0, "warnings": ["file is empty or unreadable"]}

    peak = float(np.abs(y).max())
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    loud = rms[rms > np.percentile(rms, 60)]
    quiet = rms[rms < np.percentile(rms, 20)]
    snr = float(20 * np.log10(loud.mean() / max(quiet.mean(), 1e-9))) if quiet.size else 0.0

    spec = np.abs(librosa.stft(y[: SR * 20], n_fft=2048)).mean(axis=1)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    total = float(spec.sum())
    if total <= 1e-9:
        # Silence has no bandwidth. Say so in words, not in NaN: NaN is not
        # valid JSON, so it used to surface downstream as a serialization
        # error on the voices row instead of the actual problem.
        return {"duration_s": round(dur, 2), "peak": round(peak, 3),
                "snr_db": 0.0, "bandwidth_hz": 0,
                "pitch_median_hz": None, "pitch_shift_hz": None,
                "warnings": ["recording is silent — re-export or re-record"]}
    bandwidth = float(freqs[np.searchsorted(np.cumsum(spec) / total, 0.99)])

    pitch_median = pitch_shift = None
    if dur >= 1.0:
        f0, _, _ = librosa.pyin(y, fmin=70, fmax=400, sr=SR, frame_length=2048)
        v = f0[~np.isnan(f0)]
        if v.size > 8:
            pitch_median = float(np.median(v))
            half = v.size // 2
            pitch_shift = float(abs(np.median(v[:half]) - np.median(v[half:])))

    if dur < MIN_SECONDS:
        warnings.append(f"only {dur:.1f}s — use {IDEAL_RANGE[0]:.0f}–{IDEAL_RANGE[1]:.0f}s")
    elif dur > MAX_SECONDS:
        warnings.append(f"{dur:.0f}s is long — clones worse and runs slower; trim to "
                        f"{IDEAL_RANGE[0]:.0f}–{IDEAL_RANGE[1]:.0f}s")
    elif not (IDEAL_RANGE[0] <= dur <= IDEAL_RANGE[1]):
        warnings.append(f"{dur:.1f}s works, but {IDEAL_RANGE[0]:.0f}–{IDEAL_RANGE[1]:.0f}s clones best")
    if peak >= CLIP_PEAK:
        warnings.append("audio is clipping — re-record or lower the input gain")
    if snr < MIN_SNR_DB:
        warnings.append(f"noisy (SNR {snr:.0f} dB) — background noise gets cloned too")
    if bandwidth < MIN_BANDWIDTH_HZ:
        warnings.append(f"muffled (only {bandwidth:.0f} Hz) — sample was heavily compressed")
    if pitch_shift is not None and pitch_shift > PITCH_SHIFT_WARN:
        warnings.append(f"pitch jumps {pitch_shift:.0f} Hz mid-clip — may contain two speakers")

    return {
        "duration_s": round(dur, 2),
        "peak": round(peak, 3),
        "snr_db": round(snr, 1),
        "bandwidth_hz": round(bandwidth),
        "pitch_median_hz": round(pitch_median) if pitch_median else None,
        "pitch_shift_hz": round(pitch_shift) if pitch_shift else None,
        "warnings": warnings,
    }


def build_voice(model, workdir: Path, *, language: str,
                sample_path: str | None = None, ref_text: str | None = None,
                instruct: str | None = None) -> dict:
    """Turn either a recording or an attribute description into a library voice.

    Cloned  — QC the upload, transcribe it, encode a clone prompt.
    Designed — generate one seed clip from `instruct`, then freeze THAT into a
               clone prompt. Freezing is the point: `instruct` alone re-rolls a
               different speaker on every call (measured 0.972 self-similarity
               across calls, versus 0.997 once frozen), so a designed voice
               would otherwise drift between episodes.

    Returns everything the worker needs to persist.
    """
    if instruct:
        designed = design_voice(model, instruct, language, workdir)
        sample_path = str(designed["seed_file"])
        ref_text = designed["seed_text"]
        prompt = designed["prompt"]
    elif sample_path:
        prompt = model.create_voice_clone_prompt(sample_path, ref_text=ref_text or None)
    else:
        raise ValueError("build_voice needs either sample_path or instruct")

    qc = analyse_sample(sample_path)
    if instruct:
        # A designed voice is one synthetic speaker by construction. Normal
        # intonation across a sentence trips the two-speaker pitch heuristic,
        # which is meaningful only for uploaded recordings.
        qc["warnings"] = [w for w in qc["warnings"] if "two speakers" not in w]
    # Only a truly unusable clip blocks; everything else is advice.
    blocking = qc["duration_s"] < 1.0
    qc_passed = not blocking
    used_text = getattr(prompt, "ref_text", ref_text) or ""

    prompt_file = workdir / "prompt.pt"
    prompt.save(str(prompt_file))

    line = AUDITION_TEXT.get(language, AUDITION_TEXT["English"])
    kw = {"language": language} if language else {}
    audio = model.generate(text=[line], voice_clone_prompt=[prompt], **kw)
    preview = np.asarray(audio[0], dtype=np.float32).squeeze()
    preview_file = workdir / "preview.wav"
    sf.write(preview_file, preview, SR)

    # How closely the audition matches the source timbre — a low number here
    # usually means the sample had more than one speaker in it.
    src, _ = sf.read(sample_path, dtype="float32", always_2d=False)
    if src.ndim > 1:
        src = src.mean(axis=1)
    similarity = S.cosine(S.timbre_vector(src), S.timbre_vector(preview))
    qc["self_similarity"] = round(float(similarity), 4)
    if similarity < 0.90:
        qc["warnings"].append(
            f"audition only {similarity:.2f} similar to the sample — "
            "check the clip is one speaker with no music behind it")

    return {
        "qc": qc,
        "qc_passed": qc_passed,
        "ref_text": used_text,
        "prompt_file": prompt_file,
        "preview_file": preview_file,
        "audition_line": line,
        # Designed voices have no uploaded file, so the seed clip becomes the
        # stored sample — it is what the voice actually sounds like.
        "seed_file": Path(sample_path) if instruct else None,
    }


def load_cached_prompt(model, prompt_bytes: bytes, workdir: Path):
    """Restore a VoiceClonePrompt saved by build_voice()."""
    from omnivoice.models.omnivoice import VoiceClonePrompt

    f = workdir / "cached_prompt.pt"
    f.write_bytes(prompt_bytes)
    device = getattr(model, "device", None)
    try:
        return VoiceClonePrompt.load(str(f), map_location=device or "cpu")
    except TypeError:
        return VoiceClonePrompt.load(str(f))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="QC a voice sample without a database.")
    ap.add_argument("sample")
    args = ap.parse_args()
    report = analyse_sample(args.sample)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["warnings"]:
        print("\nWarnings:")
        for w in report["warnings"]:
            print("  -", w)
    else:
        print("\nNo problems found.")
