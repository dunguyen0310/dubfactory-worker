"""Mux a dub back into the source video, keeping the music and effects bed.

What separates a professional dub from a voice-over is the background: the
original music and sound effects continue under the new voice, only the
original speech is gone. This tool does that:

    1. extract the original audio from the video (ffmpeg)
    2. split it into vocals vs music+SFX with Demucs, keep the music+SFX bed
       (no Demucs installed? fall back to using the full original mix)
    3. if that bed came back too thin to carry the background, blend a little
       of the vocals stem back into it (see BLEND below)
    4. duck the bed under the dub (gain dips while the dub speaks,
       recovers in the gaps — sidechain-style, computed sample-accurately)
    5. mix, peak-normalize, and mux back into the video; the video stream is
       copied untouched, so this is fast and lossless for the picture

    python video_dub.py --video ad.mp4 --dub dub.wav --output ad_vn.mp4

BLEND. Demucs is a *music* separation model, and its "vocals" stem is
everything voice-like — not just the dialogue. On a scene whose soundscape is
people (a crowd, an interview, a street sketch) the no_vocals bed comes back
nearly empty, because the murmur, the laughter and the shouting all left with
the lines. The dub then plays over silence, which sounds worse than not
separating at all and is the one failure mode with no obvious cause in the log.

So a copy of the vocals stem goes back into the bed at RESIDUAL_DB (-20 dB by
default). That is far too quiet to follow as speech — the duck takes another
10 dB off it while the dub speaks — but it is enough to restore the ambience
that came out with the voices. It is how an M&E track gets faked when
production never delivered one.

The blend is a rescue, so it is CONDITIONAL: it runs only when the bed is too
thin to stand on its own (SILENT_BED_DBFS / BED_DEFICIT_DB). A film with a real
music track separates cleanly, needs no rescue, and gets none — blending there
would put the original dialogue back for nothing. Measured on one such film:
blending unconditionally left the original speech 19 dB down, where leaving the
bed alone left it 30 dB down, the floor of what separation can remove.
`--no-residual` declines the rescue even when it is warranted.

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

# Wall-clock ceilings for the two external processes. Generous — htdemucs runs
# many times faster than realtime even on CPU, so a 2-hour episode fits with
# room to spare — because past these the child is wedged, not slow. Without
# them a hung Demucs held a job forever while its tick callback renewed the
# heartbeat, converting the liveness mechanism into a masking mechanism.
SEPARATE_TIMEOUT_S = 90 * 60
FFMPEG_TIMEOUT_S = 30 * 60

# Ducking: how fast the bed dives when the dub starts speaking, and how
# slowly it comes back up in pauses. Values in seconds.
ATTACK_S = 0.05
RELEASE_S = 0.40
SPEECH_RMS_DBFS = -45.0    # dub RMS above this counts as "speaking"
DUB_TARGET_DBFS = -20.0    # active-speech level the dub is normalized to
MAX_DUB_BOOST_DB = 12.0

# How much of the separated vocals stem is mixed back into the bed. Low enough
# to read as room tone, high enough to bring a crowd back. See BLEND above.
RESIDUAL_DB = -20.0

# When the separated bed counts as too thin to carry the background on its own,
# which is the only case the blend exists for. Either it is near digital silence
# outright, or it sits this far under the original mix — meaning separation took
# nearly everything, so what it took was the background as well as the voices.
SILENT_BED_DBFS = -50.0
BED_DEFICIT_DB = 20.0


class MuxError(RuntimeError):
    """A step of the mux failed.

    Raised instead of sys.exit, because this module is a library as well as a
    CLI: sys.exit raises SystemExit, which is a BaseException and not an
    Exception, so it sailed straight through the worker's `except Exception`
    guards and killed the whole worker process over one bad container. Library
    callers catch this; the CLI turns it into an exit code in main().
    """


def die(msg: str):
    sys.exit(f"error: {msg}")


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise MuxError("ffmpeg not found on PATH. Install it "
                       "(winget install Gyan.FFmpeg) and reopen the terminal.")
    return exe


def run(cmd: list[str], what: str):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=FFMPEG_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise MuxError(f"{what} did not finish in {FFMPEG_TIMEOUT_S // 60} "
                       f"minutes and was killed — the input is likely corrupt")
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        raise MuxError(f"{what} failed:\n{tail}")
    return proc


def extract_audio(video: str, out_wav: Path) -> bool:
    """Pull the video's audio track as stereo 48 kHz wav. False if it has none."""
    proc = subprocess.run(
        [ffmpeg_exe(), "-y", "-i", video, "-vn", "-ac", "2",
         "-ar", str(MASTER_SR), "-c:a", "pcm_s16le", str(out_wav)],
        capture_output=True, text=True)
    return proc.returncode == 0 and out_wav.exists() and out_wav.stat().st_size > 44


def separate_bed(orig_wav: Path, workdir: Path, log=print, tick=None):
    """Demucs two-stem split; returns ``(bed, vocals)``, or None if unavailable.

    ``bed`` is the music+SFX stem and ``vocals`` everything voice-like, which on
    dialogue-heavy material is most of the recording — hence the blend back in
    :func:`mux_dub`. ``vocals`` is None only if Demucs wrote the one stem.

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
        started = time.time()
        proc = subprocess.Popen(
            [sys.executable, "-m", "demucs", "--two-stems", "vocals",
             "-n", "htdemucs", "-o", str(out), str(orig_wav)],
            stdout=sink, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            if tick:
                tick()
            time.sleep(5)
            if time.time() - started > SEPARATE_TIMEOUT_S:
                # The tick above keeps the job's heartbeat alive, so a wedged
                # child would otherwise be held forever with every liveness
                # check reporting healthy. Kill it and degrade politely.
                proc.kill()
                log(f"  Demucs exceeded {SEPARATE_TIMEOUT_S // 60} minutes — "
                    f"killed; falling back to full-mix ducking")
                return None
    if proc.returncode != 0:
        text = demucs_log.read_text(encoding="utf-8", errors="replace")
        tail = "\n".join(text.strip().splitlines()[-6:])
        log(f"  Demucs failed, falling back to full-mix ducking:\n{tail}")
        return None
    stem_dir = out / "htdemucs" / orig_wav.stem
    bed_file = stem_dir / "no_vocals.wav"
    if not bed_file.exists():
        log("  Demucs produced no no_vocals stem — falling back")
        return None
    voc_file = stem_dir / "vocals.wav"
    return (_read_at_master(bed_file),
            _read_at_master(voc_file) if voc_file.exists() else None)


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    import librosa
    if x.ndim == 1:
        return librosa.resample(x, orig_sr=sr_from, target_sr=sr_to)
    return np.stack([librosa.resample(np.ascontiguousarray(x[:, c]),
                                      orig_sr=sr_from, target_sr=sr_to)
                     for c in range(x.shape[1])], axis=1)


def _read_at_master(path: Path) -> np.ndarray:
    """Read a wav as 2-D float32 at MASTER_SR."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return resample(x, sr, MASTER_SR) if sr != MASTER_SR else x


def dbfs(x: np.ndarray) -> float:
    """RMS level of `x` in dBFS, floored at -99 so it logs and stores cleanly."""
    if not x.size:
        return -99.0
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return max(-99.0, 20.0 * float(np.log10(max(rms, 1e-12))))


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
            residual_db: float | None = RESIDUAL_DB,
            keep_audio: str | None = None, log=print, tick=None) -> dict:
    """Lay `dub` over `video`'s music/effects bed and write `output`.

    `residual_db` is how much of the separated vocals stem goes back into the
    bed to restore ambience; None keeps the bed pure. It does nothing without a
    separation, because the unseparated bed is the full mix already.

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
    bed = residual = None
    separated = False
    if has_audio:
        if separate:
            stems = separate_bed(orig_wav, work, log=log, tick=tick)
            if stems is not None:
                bed, residual = stems
                separated = True
        if bed is None:
            bed, _ = sf.read(orig_wav, dtype="float32", always_2d=True)
            log("using the full original mix as the bed "
                "(original speech remains, ducked)")
        else:
            log("bed = music + effects only (original speech removed)")

    # ---- blend the ambience back ----------------------------------------
    # The blend is a RESCUE, not a garnish, so it is conditional. Where the
    # separation left a real music/effects bed there is nothing to rescue, and
    # blending would only put the original dialogue back for no gain. It runs
    # only when the bed came back too thin to carry the background at all.
    bed_dbfs = dbfs(bed) if bed is not None else None
    orig_dbfs = dbfs(_read_at_master(orig_wav)) if has_audio else None

    # Two ways to be dead, because an absolute floor alone misjudges a quiet
    # source: a bed at -48 dBFS is healthy under a -45 dBFS mix and hopeless
    # under a -12 dBFS one. The deficit catches that; the floor catches digital
    # silence, where the deficit would divide a near-zero by a near-zero.
    deficit = (orig_dbfs - bed_dbfs) if (separated and orig_dbfs is not None) else 0.0
    silent_bed = bool(separated and bed_dbfs is not None
                      and (bed_dbfs <= SILENT_BED_DBFS
                           or deficit >= BED_DEFICIT_DB))
    applied_residual = None
    if separated:
        log(f"bed {bed_dbfs:.0f} dBFS, {deficit:.0f} dB below the original mix")
        if not silent_bed:
            log("  bed is intact — no ambience blend needed")
        elif residual is None:
            log("  bed is thin, but there is no vocals stem to blend from")
        elif residual_db is None:
            log("  bed is thin and the blend is off — the dub will play over "
                "near-silence; drop --no-residual, or --no-separate to keep "
                "the whole mix")
        else:
            n_r = min(bed.shape[0], residual.shape[0])
            bed[:n_r] += residual[:n_r] * 10 ** (residual_db / 20)
            applied_residual = float(residual_db)
            log(f"  bed is thin — blending ambience back at {residual_db:.0f} "
                f"dB (bed {bed_dbfs:.0f} -> {dbfs(bed):.0f} dBFS)")

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
        "residual_db": (None if applied_residual is None
                        else round(applied_residual, 1)),
        "bed_dbfs": None if bed_dbfs is None else round(bed_dbfs, 1),
        "bed_deficit_db": round(float(deficit), 1) if separated else None,
        "silent_bed": silent_bed,
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
    p.add_argument("--residual-db", type=float, default=RESIDUAL_DB,
                   help="how much of the separated vocals stem is blended back "
                        "into the bed, restoring crowd and room tone "
                        f"(default {RESIDUAL_DB:.0f})")
    p.add_argument("--no-residual", action="store_true",
                   help="keep the separated bed pure — blend nothing back")
    p.add_argument("--keep-audio", help="also save the mixed track as this .wav")
    args = p.parse_args()

    try:
        mux_dub(args.video, args.dub, args.output,
                duck_db=args.duck_db, bed_gain_db=args.bed_gain_db,
                separate=not args.no_separate,
                residual_db=None if args.no_residual else args.residual_db,
                keep_audio=args.keep_audio)
    except (FileNotFoundError, MuxError) as e:
        die(str(e))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
