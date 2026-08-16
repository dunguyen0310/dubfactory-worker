"""Mux a dub back into the source video, keeping the music and effects bed.

What separates a professional dub from a voice-over is the background: the
original music and sound effects continue under the new voice, only the
original speech is gone. This tool does that:

    1. extract the original audio from the video (ffmpeg)
    2. split it into vocals vs music+SFX with Demucs, keep the music+SFX bed
       (no Demucs installed? fall back to using the full original mix)
    3. duck the bed under the dub (gain dips while the dub speaks,
       recovers in the gaps — sidechain-style, computed sample-accurately)
    4. mix, peak-normalize, and mux back into the video; the video stream is
       copied untouched, so this is fast and lossless for the picture

    python video_dub.py --video ad.mp4 --dub dub.wav --output ad_vn.mp4

Demucs is optional but recommended: without it the original voices are still
underneath, just pushed down hard. Install (Linux/Colab, or Windows with a
matching torchaudio):

    pip install torchaudio==<your torch version> demucs

ffmpeg must be on PATH.
"""

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

MASTER_SR = 48000          # video-standard rate the mix is built at

# Ducking: how fast the bed dives when the dub starts speaking, and how
# slowly it comes back up in pauses. Values in seconds.
ATTACK_S = 0.05
RELEASE_S = 0.40
SPEECH_RMS_DBFS = -45.0    # dub RMS above this counts as "speaking"
DUB_TARGET_DBFS = -20.0    # active-speech level the dub is normalized to
MAX_DUB_BOOST_DB = 12.0


def die(msg: str):
    sys.exit(f"error: {msg}")


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        die("ffmpeg not found on PATH. Install it (winget install Gyan.FFmpeg) "
            "and reopen the terminal.")
    return exe


def run(cmd: list[str], what: str):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        die(f"{what} failed:\n{tail}")
    return proc


def extract_audio(video: str, out_wav: Path) -> bool:
    """Pull the video's audio track as stereo 48 kHz wav. False if it has none."""
    proc = subprocess.run(
        [ffmpeg_exe(), "-y", "-i", video, "-vn", "-ac", "2",
         "-ar", str(MASTER_SR), "-c:a", "pcm_s16le", str(out_wav)],
        capture_output=True, text=True)
    return proc.returncode == 0 and out_wav.exists() and out_wav.stat().st_size > 44


def separate_bed(orig_wav: Path, workdir: Path, log=print, tick=None):
    """Demucs two-stem split; returns the music+SFX bed, or None if unavailable.

    Runs the demucs CLI in a subprocess rather than importing an API: the CLI
    is stable across every released version, the api module is not.

    `tick`, if given, is called every few seconds while Demucs runs. A full
    episode separates for many silent minutes, and the caller may need to
    prove it is still alive during them — the worker's stall sweep requeues
    any job quiet for 15, and handing a live separation to a second worker
    would render the same episode twice.
    """
    if importlib.util.find_spec("demucs") is None:
        log("Demucs not installed — ducking the full original mix instead "
            "(pip install demucs to remove the original voices)")
        return None
    log("separating vocals from music/SFX (Demucs htdemucs — first run "
        "downloads the model, ~300 MB) ...")
    out = workdir / "demucs"
    # Output goes to a file rather than a pipe: nobody drains a pipe while we
    # sit in the poll loop, and a full pipe buffer deadlocks the child.
    demucs_log = workdir / "demucs.log"
    with open(demucs_log, "w", encoding="utf-8", errors="replace") as sink:
        proc = subprocess.Popen(
            [sys.executable, "-m", "demucs", "--two-stems", "vocals",
             "-n", "htdemucs", "-o", str(out), str(orig_wav)],
            stdout=sink, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            if tick:
                tick()
            time.sleep(5)
    if proc.returncode != 0:
        text = demucs_log.read_text(encoding="utf-8", errors="replace")
        tail = "\n".join(text.strip().splitlines()[-6:])
        log(f"  Demucs failed, falling back to full-mix ducking:\n{tail}")
        return None
    bed_file = out / "htdemucs" / orig_wav.stem / "no_vocals.wav"
    if not bed_file.exists():
        log("  Demucs produced no no_vocals stem — falling back")
        return None
    bed, sr = sf.read(bed_file, dtype="float32", always_2d=True)
    if sr != MASTER_SR:
        bed = resample(bed, sr, MASTER_SR)
    return bed


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    import librosa
    if x.ndim == 1:
        return librosa.resample(x, orig_sr=sr_from, target_sr=sr_to)
    return np.stack([librosa.resample(np.ascontiguousarray(x[:, c]),
                                      orig_sr=sr_from, target_sr=sr_to)
                     for c in range(x.shape[1])], axis=1)


def speech_gain_curve(dub: np.ndarray, duck_gain: float) -> np.ndarray:
    """Per-sample gain for the bed: dips to duck_gain while the dub speaks.

    RMS per 10 ms hop -> active flags -> attack/release smoothing, then
    interpolated back to sample rate so the gain never steps audibly.
    """
    hop = MASTER_SR // 100
    n_hops = max(1, int(np.ceil(dub.size / hop)))
    padded = np.pad(dub, (0, n_hops * hop - dub.size))
    rms = np.sqrt(np.mean(padded.reshape(n_hops, hop) ** 2, axis=1))
    active = rms > 10 ** (SPEECH_RMS_DBFS / 20)

    target = np.where(active, duck_gain, 1.0)
    attack = 1 - np.exp(-1 / (ATTACK_S * 100))    # per-hop smoothing coefs
    release = 1 - np.exp(-1 / (RELEASE_S * 100))
    g = np.empty(n_hops, dtype=np.float32)
    cur = 1.0
    for i, t in enumerate(target):
        cur += (t - cur) * (attack if t < cur else release)
        g[i] = cur

    centers = np.arange(n_hops) * hop + hop / 2
    return np.interp(np.arange(dub.size), centers, g).astype(np.float32)


def normalize_dub(dub: np.ndarray) -> np.ndarray:
    """Bring the dub's spoken parts to a consistent level."""
    hop = MASTER_SR // 100
    n_hops = max(1, int(np.ceil(dub.size / hop)))
    padded = np.pad(dub, (0, n_hops * hop - dub.size))
    rms = np.sqrt(np.mean(padded.reshape(n_hops, hop) ** 2, axis=1))
    speaking = rms[rms > 10 ** (SPEECH_RMS_DBFS / 20)]
    if not speaking.size:
        return dub
    gain = 10 ** (DUB_TARGET_DBFS / 20) / max(float(np.mean(speaking)), 1e-9)
    gain = min(gain, 10 ** (MAX_DUB_BOOST_DB / 20))
    return dub * gain


def mux_dub(video: str, dub: str, output: str, *, duck_db: float | None = None,
            bed_gain_db: float = 0.0, separate: bool = True,
            keep_audio: str | None = None, log=print, tick=None) -> dict:
    """Lay `dub` over `video`'s music/effects bed and write `output`.

    Importable entry point — worker.py calls this after assembling a dub.
    Returns a summary dict suitable for storing as job QC evidence.
    """
    if not Path(video).exists():
        raise FileNotFoundError(f"video not found: {video}")
    if not Path(dub).exists():
        raise FileNotFoundError(f"dub not found: {dub}")

    work = Path(tempfile.mkdtemp(prefix="videodub_"))

    # ---- original audio --------------------------------------------------
    orig_wav = work / "original.wav"
    has_audio = extract_audio(video, orig_wav)
    if not has_audio:
        log("video has no audio track — output will be dub only")

    # ---- dub -------------------------------------------------------------
    voice, dub_sr = sf.read(dub, dtype="float32", always_2d=False)
    if voice.ndim > 1:
        voice = voice.mean(axis=1)
    voice = resample(voice, dub_sr, MASTER_SR)
    voice = normalize_dub(voice)

    # ---- bed -------------------------------------------------------------
    bed = None
    separated = False
    if has_audio:
        if separate:
            bed = separate_bed(orig_wav, work, log=log, tick=tick)
            separated = bed is not None
        if bed is None:
            bed, _ = sf.read(orig_wav, dtype="float32", always_2d=True)
            log("using the full original mix as the bed "
                "(original speech remains, ducked)")
        else:
            log("bed = music + effects only (original speech removed)")

    if duck_db is None:
        duck_db = 10.0 if separated else 18.0
    duck_gain = 10 ** (-abs(duck_db) / 20)

    # ---- mix -------------------------------------------------------------
    n = voice.size if bed is None else max(voice.size, bed.shape[0])
    mix = np.zeros((n, 2), dtype=np.float32)
    if bed is not None:
        gain_curve = speech_gain_curve(
            np.pad(voice, (0, n - voice.size)), duck_gain)
        bed_level = 10 ** (bed_gain_db / 20)
        seg = bed[:n] * gain_curve[:bed.shape[0], None] * bed_level
        mix[:seg.shape[0]] += seg
        log(f"ducking bed {duck_db:.0f} dB under speech "
            f"(attack {ATTACK_S*1000:.0f} ms, release {RELEASE_S*1000:.0f} ms)")
    mix[:voice.size, 0] += voice
    mix[:voice.size, 1] += voice

    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.98:
        mix *= 0.98 / peak

    mix_wav = work / "mix.wav"
    sf.write(mix_wav, mix, MASTER_SR)
    if keep_audio:
        sf.write(keep_audio, mix, MASTER_SR)
        log(f"wrote {keep_audio}")

    # ---- mux -------------------------------------------------------------
    run([ffmpeg_exe(), "-y", "-i", video, "-i", str(mix_wav),
         "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-movflags", "+faststart", output],
        "muxing")

    dub_len = voice.size / MASTER_SR
    bed_len = bed.shape[0] / MASTER_SR if bed is not None else 0.0
    overrun = max(0.0, dub_len - bed_len) if bed is not None else 0.0
    if overrun > 0.5:
        log(f"NOTE: the dub runs {overrun:.1f}s past the video "
            f"(natural-mode overrun). The audio track carries it, but the "
            f"picture ends first — re-cut the video with the corrected .srt "
            f"or re-render in Fit mode for an exact match.")
    log(f"dub length {dub_len:.1f}s, bed "
        + (f"{bed_len:.1f}s" if bed is not None else "none")
        + f", separation {'ON' if separated else 'off'}")

    return {
        "separated": separated,
        "duck_db": round(float(duck_db), 1),
        "dub_seconds": round(dub_len, 2),
        "video_seconds": round(bed_len, 2),
        "overrun_seconds": round(overrun, 2),
        "had_audio": has_audio,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="source video (any ffmpeg format)")
    p.add_argument("--dub", required=True, help="dub.wav from srt_dub.py / the worker")
    p.add_argument("--output", required=True, help="output video file")
    p.add_argument("--duck-db", type=float, default=None,
                   help="how far the bed drops while the dub speaks "
                        "(default: 10 dB with separation, 18 dB without)")
    p.add_argument("--bed-gain-db", type=float, default=0.0,
                   help="overall bed level adjustment")
    p.add_argument("--no-separate", action="store_true",
                   help="skip Demucs even if installed (duck the full mix)")
    p.add_argument("--keep-audio", help="also save the mixed track as this .wav")
    args = p.parse_args()

    try:
        mux_dub(args.video, args.dub, args.output,
                duck_db=args.duck_db, bed_gain_db=args.bed_gain_db,
                separate=not args.no_separate, keep_audio=args.keep_audio)
    except FileNotFoundError as e:
        die(str(e))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
