"""VieNeu Studio worker — the second engine, drained separately from worker.py.

Speaks the `vieneu_jobs` / `vieneu_voices` queues with VieNeu-TTS v3 Turbo
(huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo). Nothing here imports the dub
pipeline and nothing there imports this: the two share a Supabase project and
nothing else, so a broken OmniVoice install does not stop the studio and a
broken VieNeu install does not stop an episode.

Run it alongside worker.py or instead of it. **It does not need a GPU.** The
default backend is ONNX Runtime on CPU, which is torch-free and still faster
than realtime, so this can live on the machine that is already open rather than
on a Colab session somebody has to remember to start.

    pip install vieneu supabase soundfile numpy
    python vieneu_worker.py

Three job kinds, all claimed from one loop:

    speak    a script, chunked by the app, rendered and joined into one file
    batch    many lines, one file each, plus a zip and a joined audition
    denoise  no synthesis at all — clean a recording and hand it back

Plus a voice queue: a `vieneu_voices` row with a reference clip is enrolled and
given an audition, which is the only honest way to find out whether a clone
worked.

`vieneu_jobs.settings` is passed through: any key in MODEL_KWARGS below reaches
the model, so a job can set `top_k`, `silence_p` or `apply_watermark` without a
worker change. Note that v3 Turbo **watermarks its output by default**
(`apply_watermark=True`); this worker leaves that alone.

Credentials — one of these two sets, same as worker.py:

    A. Self-hosted / own Supabase project
        SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

    B. Lovable Cloud (no service_role key is issued)
        SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, WORKER_EMAIL, WORKER_PASSWORD
       The worker signs in as a staff account and works under RLS, so it sees
       only that account's jobs. Queue from the same login.

Other environment:
    VIENEU_BACKEND        onnx (default, CPU) | pytorch (CUDA)
    VIENEU_PRECISION      int8 (default) | fp32
    VIENEU_DEVICE         cuda:0 — only consulted on the pytorch backend
    VIENEU_POLL_SECONDS   idle poll interval (default 5)
    VIENEU_WORKER_ID      presence row id (default: hostname-pid)
    VIENEU_WORKER_LABEL   name shown in the web app

Run:
    python vieneu_worker.py                       # loop until stopped
    python vieneu_worker.py --once                # drain both queues then exit
    python vieneu_worker.py --runtime-minutes 200 # stop before a Colab session ends
    python vieneu_worker.py --list-voices         # print the model's presets and exit
"""

import argparse
import io
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

# Every preset voice is named in Vietnamese and so is most of what this renders,
# but a Windows console is cp1252 by default and `print("Phạm Tuyên")` raises
# UnicodeEncodeError there. The worker would then die inside its *logging*,
# after the work was done — so the stream is widened before anything prints.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass                      # already UTF-8, or a stream that cannot be set

POLL = float(os.environ.get("VIENEU_POLL_SECONDS", "5"))
BACKEND = os.environ.get("VIENEU_BACKEND", "onnx")
PRECISION = os.environ.get("VIENEU_PRECISION", "int8")
DEVICE = os.environ.get("VIENEU_DEVICE", "cuda:0")

BUCKET_IN = "vieneu"
BUCKET_OUT = "vieneu-outputs"

# What an audition says. Vietnamese with tone marks, an English clause to prove
# the code-switching, and a number — the three things a listener is checking.
AUDITION_TEXT = (
    "Xin chào, đây là giọng đọc thử của VieNeu — real time, forty eight kilohertz."
)

_model = None
_model_info: dict[str, str] = {}
# Cloned voices enrolled in *this* process. `add_voice` is per-session state, so
# the map is rebuilt on every start; the DB row is the durable half.
_enrolled: dict[str, str] = {}
_presets: set[str] = set()
_stats = {"jobs": 0, "voices": 0, "lines": 0, "audio_s": 0.0}


# ----------------------------------------------------------------- supabase

_auth = {"email": None, "password": None, "at": 0.0}


def get_client():
    """Connect either as service_role, or as a signed-in user.

    Lovable Cloud manages its Supabase project and does not hand out the
    service_role key, so the user path is the one that works there. The RLS
    policies in 20260821_vieneu_studio.sql grant `authenticated` what a worker
    needs; it simply only sees that account's rows.
    """
    try:
        from supabase import create_client
    except ImportError:
        sys.exit("pip install supabase")

    url = os.environ.get("SUPABASE_URL")
    if not url:
        sys.exit("Set SUPABASE_URL.")

    service = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if service:
        print("auth: service_role (sees all users' jobs)", flush=True)
        return create_client(url, service)

    anon = (os.environ.get("SUPABASE_ANON_KEY")
            or os.environ.get("SUPABASE_PUBLISHABLE_KEY"))
    email = os.environ.get("WORKER_EMAIL")
    password = os.environ.get("WORKER_PASSWORD")
    if not (anon and email and password):
        sys.exit(
            "No usable credentials.\n"
            "  Either: SUPABASE_SERVICE_ROLE_KEY  (self-hosted Supabase)\n"
            "  Or    : SUPABASE_PUBLISHABLE_KEY + WORKER_EMAIL + WORKER_PASSWORD\n"
            "          (Lovable Cloud — sign in as a staff account)"
        )

    sb = create_client(url, anon)
    res = sb.auth.sign_in_with_password({"email": email, "password": password})
    if not getattr(res, "user", None):
        sys.exit("Sign-in failed — check WORKER_EMAIL / WORKER_PASSWORD.")
    _auth.update(email=email, password=password, at=time.time())
    print(f"auth: signed in as {email} (only this account's jobs are visible)",
          flush=True)
    return sb


# Supabase access tokens last an hour; refreshing at half that leaves room for
# a failed attempt and a slow retry before anything actually expires.
TOKEN_MAX_AGE = 30 * 60


def refresh_auth(sb, force: bool = False):
    """Re-sign-in before the JWT expires. Cheap to call often.

    A 500-line batch can outlive a token, and the moment it would expire is
    during the final uploads — after every second of compute has been spent.
    worker.py lost a 411-cue episode to exactly that.
    """
    if not _auth["email"]:
        return                        # service_role key — never expires
    age = time.time() - _auth["at"]
    if not force and age < TOKEN_MAX_AGE:
        return
    try:
        sb.auth.sign_in_with_password({"email": _auth["email"],
                                       "password": _auth["password"]})
        _auth["at"] = time.time()
        print(f"  (auth token refreshed after {age / 60:.0f} min)", flush=True)
    except Exception as e:
        # Leave the timestamp alone so the next call retries rather than
        # waiting another full interval on a token that is already stale.
        print(f"  (token refresh failed, will retry: {e})", flush=True)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def set_job(sb, job_id, **fields):
    sb.table("vieneu_jobs").update(fields).eq("id", job_id).execute()


def set_progress(sb, job_id, **fields):
    """Progress writes must never kill a render.

    Whatever this reports on is already uploaded; losing a counter refresh to a
    network blip costs nothing, losing the job to it costs the whole render.
    """
    try:
        set_job(sb, job_id, **fields)
    except Exception as e:
        print(f"  (progress update failed: {e})", flush=True)


def download(sb, bucket, path) -> bytes:
    return sb.storage.from_(bucket).download(path)


def upload(sb, bucket, path, data: bytes, content_type: str):
    """Upload, retrying once on a fresh token.

    The last upload of a job runs at the exact moment the token is oldest,
    which is also the most expensive possible moment to fail. One
    refresh-and-retry covers both an expired token and a transient blip; a
    second failure is a real outage and raises.
    """
    try:
        sb.storage.from_(bucket).upload(
            path, data, {"content-type": content_type, "upsert": "true"})
    except Exception as e:
        print(f"  (upload {path} failed, retrying once: {e})", flush=True)
        refresh_auth(sb, force=True)
        sb.storage.from_(bucket).upload(
            path, data, {"content-type": content_type, "upsert": "true"})


# ------------------------------------------------------------- scratch space

_SCRATCH: list[Path] = []


def scratch(prefix: str) -> Path:
    d = Path(tempfile.mkdtemp(prefix=prefix))
    _SCRATCH.append(d)
    return d


def _sweep_scratch():
    while _SCRATCH:
        shutil.rmtree(_SCRATCH.pop(), ignore_errors=True)


# ----------------------------------------------------------------- presence

BEAT_SECONDS = 20
WORKER_ID = (os.environ.get("VIENEU_WORKER_ID")
             or f"vieneu-{platform.node() or 'worker'}-{os.getpid()}")
_beat = {"at": 0.0, "on": True, "fails": 0}


def worker_label():
    if os.environ.get("VIENEU_WORKER_LABEL"):
        return os.environ["VIENEU_WORKER_LABEL"]
    if os.environ.get("COLAB_RELEASE_TAG") or os.path.isdir("/content/drive"):
        return "Colab"
    if os.environ.get("RUNPOD_POD_ID"):
        return f"RunPod {os.environ['RUNPOD_POD_ID'][:8]}"
    return platform.node() or "worker"


def beat(sb, status, job_id=None, detail=None, force=False):
    """Write this worker's presence row. Rate-limited unless forced.

    Presence is optional and must never be the thing that breaks a render, so
    three consecutive failures turn it off for the session — that distinguishes
    "the migration was never run" from one dropped request.
    """
    if not _beat["on"]:
        return
    if not force and time.time() - _beat["at"] < BEAT_SECONDS:
        return
    try:
        sb.table("vieneu_workers").upsert({
            "id": WORKER_ID, "label": worker_label(),
            "device": _model_info.get("device", "cpu"),
            "backend": _model_info.get("backend", BACKEND),
            "precision": _model_info.get("precision", PRECISION),
            "status": status, "job_id": job_id, "detail": detail,
            "last_seen_at": now_iso(),
            "stats": {"jobs": _stats["jobs"], "voices": _stats["voices"],
                      "lines": _stats["lines"],
                      "audio_min": round(_stats["audio_s"] / 60, 1)},
        }).execute()
        _beat["at"] = time.time()
        _beat["fails"] = 0
    except Exception as e:
        _beat["fails"] += 1
        if _beat["fails"] >= 3:
            _beat["on"] = False
            print(f"  (engine status off — vieneu_workers unavailable: {e})",
                  flush=True)
        else:
            print(f"  (presence write failed, will retry: {e})", flush=True)


_touch = {"at": 0.0}


def touch(sb, job, detail=None):
    """Prove a long job is alive. Rate-limited, renews the token, never raises.

    `requeue_stalled_vieneu()` hands a job to another worker after ten minutes
    of silence, and a 500-line batch is easily longer than that.
    """
    if time.time() - _touch["at"] < 30:
        return
    _touch["at"] = time.time()
    try:
        refresh_auth(sb)
        set_job(sb, job["id"], heartbeat_at=now_iso())
        beat(sb, "busy", job_id=job["id"], detail=detail or job.get("title"))
    except Exception as e:
        print(f"  (heartbeat failed: {e})", flush=True)


# -------------------------------------------------------------------- model

def get_model():
    """Load VieNeu once, and record how it was actually configured.

    The requested backend is not always the one you get — `pytorch` needs torch
    and a CUDA device, and asking for it on a CPU box should degrade to ONNX
    rather than fail every job. What the model ends up as is written to the
    presence row, so the app shows the truth rather than the request.
    """
    global _model
    if _model is not None:
        return _model

    try:
        from vieneu import Vieneu
    except ImportError:
        sys.exit(
            "pip install vieneu\n"
            "  CPU (default, torch-free):  pip install vieneu\n"
            "  GPU (CUDA >= 12.8):         pip install torch==2.8.0 torchaudio==2.8.0 "
            "--index-url https://download.pytorch.org/whl/cu128 && "
            'pip install "transformers==4.57.6" vieneu'
        )

    backend, device = BACKEND, "cpu"
    if backend == "pytorch":
        try:
            import torch
            if torch.cuda.is_available():
                device = DEVICE
            else:
                print("  (no CUDA device — falling back to the ONNX/CPU backend)",
                      flush=True)
                backend = "onnx"
        except ImportError:
            print("  (torch not installed — falling back to the ONNX/CPU backend)",
                  flush=True)
            backend = "onnx"

    print(f"loading VieNeu-TTS v3 Turbo — {backend} / {PRECISION} / {device} ...",
          flush=True)
    t0 = time.time()
    _model = Vieneu(backend=backend, precision=PRECISION, device=device)
    _model_info.update(backend=backend, precision=PRECISION, device=device)

    # The names the model will actually accept. `list_preset_voices()` returns
    # (label, voice_id) where the label carries a description — only the id is
    # what `voice=` takes, so only the id belongs in this set.
    try:
        _presets.update(vid for _, vid in _model.list_preset_voices())
    except Exception as e:
        print(f"  (could not list preset voices: {e})", flush=True)

    print(f"  loaded in {time.time() - t0:.0f}s — {len(_presets)} preset names",
          flush=True)
    return _model


def known_preset(name: str) -> bool:
    """Unknown until the model has been asked; then authoritative.

    An empty set means `list_preset_voices()` failed, and refusing every job on
    that basis would be worse than letting the model refuse the name itself.
    """
    return not _presets or name in _presets


def supported(fn, *names) -> dict:
    """Which of these keyword arguments the installed build actually takes.

    The library is young and its signatures move — `temperature` is documented
    as a recommendation rather than shown in a call, and `denoise` appears on
    `infer` but not in any `add_voice` example. Asking the signature is how an
    optional knob stays optional: a build without it renders at the default
    instead of raising TypeError on every single line.

    Deliberately not a try/except around the call, which would also swallow a
    TypeError raised from *inside* the model and retry a call that had a real
    problem.
    """
    try:
        import inspect
        params = inspect.signature(fn).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return {n: True for n in names}
        return {n: n in params for n in names}
    except (TypeError, ValueError):
        # A C-extension or wrapped callable with no introspectable signature.
        # Passing nothing optional is the safe read.
        return {n: False for n in names}


# ------------------------------------------------------------------- audio

# Generation knobs `infer`, `infer_stream` and `infer_batch` all accept. Copied
# from `settings` when present, so a job can reach any of them without a worker
# change — the same open contract `jobs.settings` gives the dub pipeline.
#
# `apply_watermark` is in the list and defaults to True in the model: v3 Turbo
# watermarks its output unless told otherwise. Leaving it on is the default here
# too; a job that sets it false has made that choice deliberately.
MODEL_KWARGS = (
    "temperature", "top_k", "top_p", "max_new_frames", "repetition_penalty",
    "repetition_window", "max_chars", "silence_p", "crossfade_p",
    "apply_watermark", "use_ref_codes",
)


def model_kwargs(settings: dict) -> dict:
    """The subset of `settings` that means something to the model."""
    return {k: settings[k] for k in MODEL_KWARGS
            if settings.get(k) is not None}


def render(model, text: str, voice: str, work: Path, tag: str,
           gen: dict):
    """Speak one line and return (wav bytes, samples, sample rate).

    Written through the model's own `save()` and read back rather than encoded
    here: `save` knows the dtype and channel layout the model emits, and
    guessing at them is how a render comes out as static.
    """
    audio = model.infer(text, voice=voice, **gen)

    path = work / f"{tag}.wav"
    model.save(audio, str(path))
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return path.read_bytes(), data, sr


def encode(samples: np.ndarray, sr: int, fmt: str) -> tuple[bytes, str, str]:
    """Serialise a finished array. Falls back to WAV if MP3 is unavailable.

    libsndfile only grew MP3 writing in 1.1, and the version that ships with a
    given wheel is not something a job should fail on — the audio is already
    rendered, and a WAV nobody asked for still plays.
    """
    if fmt == "mp3":
        try:
            buf = io.BytesIO()
            sf.write(buf, samples, sr, format="MP3")
            return buf.getvalue(), "mp3", "audio/mpeg"
        except Exception as e:
            print(f"  (mp3 unavailable, writing wav instead: {e})", flush=True)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue(), "wav", "audio/wav"


def join(clips: list[np.ndarray], sr: int, gap_ms: int) -> np.ndarray:
    """Concatenate chunks with a gap, so a joined read breathes.

    Butting chunks together sounds like an edit because it is one: each was
    generated with its own onset and decay, and back-to-back they collide. A
    fifth of a second is roughly the pause a reader takes between sentences.
    """
    if not clips:
        return np.zeros(0, dtype=np.float32)
    silence = np.zeros(int(sr * max(gap_ms, 0) / 1000), dtype=np.float32)
    out: list[np.ndarray] = []
    for i, c in enumerate(clips):
        if i:
            out.append(silence)
        out.append(np.asarray(c, dtype=np.float32))
    return np.concatenate(out)


def dbfs(samples: np.ndarray) -> float:
    """RMS level in dBFS. -inf guarded, because digital silence is a real input."""
    x = np.asarray(samples, dtype=np.float64).ravel()
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(x * x)))
    return 20 * np.log10(rms) if rms > 1e-9 else -120.0


# ------------------------------------------------------------ voice library

def build_voice(sb, voice):
    """Enroll a cloned voice and render its audition.

    The audition is the deliverable, not a nicety: cloning is instant and never
    reports a quality score, so the only way to know whether a reference clip
    worked is to hear the result. A voice that reaches `ready` without one
    would be a voice nobody can evaluate until they have spent a job on it.
    """
    print(f"\n=== voice {voice['id']} — {voice['name']} ===", flush=True)
    work = scratch("vieneu-voice-")

    if not voice.get("sample_path"):
        raise RuntimeError("No reference recording was uploaded.")

    ref = work / "reference.wav"
    ref.write_bytes(download(sb, BUCKET_IN, voice["sample_path"]))

    # Measured before the model sees it, so a rejected clip is explained in the
    # terms the person who uploaded it can act on — "too short", not "failed".
    qc: dict[str, object] = {}
    warnings: list[str] = []
    try:
        info = sf.info(str(ref))
        data, sr = sf.read(str(ref), dtype="float32", always_2d=False)
        qc["duration_s"] = round(float(info.duration), 2)
        qc["sample_rate"] = int(info.samplerate)
        qc["peak_dbfs"] = round(float(dbfs(data)), 1)
        if info.duration < 2.5:
            warnings.append("under 3s — the model has little to work from")
        if info.duration > 12:
            warnings.append("over 8s is trimmed by the model; the tail is unused")
        if info.channels > 1:
            warnings.append(f"{info.channels} channels — mixed down before cloning")
        if qc["peak_dbfs"] < -35:
            warnings.append("very quiet — normalise it and re-upload")
    except Exception as e:
        # A file soundfile cannot open is very likely one the model cannot
        # either, but let the model be the one to say so.
        warnings.append(f"could not probe the file: {e}")

    if warnings:
        qc["warnings"] = warnings
    sb.table("vieneu_voices").update({"qc": qc}).eq("id", voice["id"]).execute()

    model = get_model()
    key = enrol_key(voice["id"])
    enrol(model, key, ref, bool(voice.get("denoise", True)))
    _enrolled[voice["id"]] = key

    wav, samples, sr = render(model, AUDITION_TEXT, key, work, "audition", {})
    preview = f"{voice['owner_id']}/{voice['id']}/audition.wav"
    upload(sb, BUCKET_IN, preview, wav, "audio/wav")

    sb.table("vieneu_voices").update({
        "status": "ready", "preview_path": preview, "error": None, "qc": qc,
    }).eq("id", voice["id"]).execute()

    _stats["voices"] += 1
    _stats["audio_s"] += len(samples) / sr
    print(f"  ready — {len(samples) / sr:.1f}s audition", flush=True)


def enrol(model, key: str, ref: Path, denoise: bool):
    """Register a cloned voice under `key`, denoising the reference if asked.

    `denoise` is passed only when the installed build takes it. Cleaning the
    reference is what stops room noise being cloned along with the voice, so
    losing the flag is a real quality difference — but it is not worth failing
    every clone over, and the default is on anyway.
    """
    kwargs = {}
    if supported(model.add_voice, "denoise")["denoise"]:
        kwargs["denoise"] = denoise
    elif not denoise:
        print("  (this build of vieneu has no add_voice(denoise=...); "
              "the reference is cleaned regardless)", flush=True)
    model.add_voice(key, str(ref), **kwargs)


def enrol_key(voice_id: str) -> str:
    """A registration name that cannot collide with a preset or another clone.

    Two people can both name a voice "Chị Trân", and either could name one
    "Adam". The row id cannot be either.
    """
    return f"db-{voice_id}"


def voice_for(sb, model, preset: str | None, voice_id: str | None, work: Path) -> str:
    """Resolve a job's or line's voice to a name the model will accept.

    Cloned voices are enrolled lazily and cached for the session: `add_voice`
    is process state, so a restart has to redo it, and doing it per line would
    re-encode the same reference once per line.
    """
    if preset:
        if not known_preset(preset):
            raise RuntimeError(
                f'Unknown preset voice "{preset}". The model knows: '
                + ", ".join(sorted(_presets)[:8]) + " …"
            )
        return preset

    if not voice_id:
        raise RuntimeError("The job names no voice.")

    if voice_id in _enrolled:
        return _enrolled[voice_id]

    rows = (sb.table("vieneu_voices").select("*").eq("id", voice_id)
            .limit(1).execute().data)
    if not rows:
        raise RuntimeError("The cloned voice this job uses has been deleted.")
    v = rows[0]
    if not v.get("consent_confirmed"):
        raise RuntimeError(f'"{v["name"]}" has no consent recorded, so it cannot be used.')
    if not v.get("sample_path"):
        raise RuntimeError(f'"{v["name"]}" has no reference recording.')

    ref = work / f"ref-{voice_id}.wav"
    ref.write_bytes(download(sb, BUCKET_IN, v["sample_path"]))
    key = enrol_key(voice_id)
    enrol(model, key, ref, bool(v.get("denoise", True)))
    _enrolled[voice_id] = key
    return key


# ------------------------------------------------------------------- jobs

def denoise_job(sb, job):
    """Clean a recording. No model voice, no lines, no synthesis."""
    if not job.get("source_path"):
        raise RuntimeError("No file was uploaded to clean.")

    work = scratch("vieneu-denoise-")
    src = work / "input"
    src.write_bytes(download(sb, BUCKET_IN, job["source_path"]))
    before, _ = sf.read(str(src), dtype="float32", always_2d=False)

    model = get_model()
    out = work / "clean.wav"
    # Returns 44.1 kHz rather than the 48 the synthesiser emits, so the rate is
    # read back from the file rather than assumed.
    model.denoise(str(src), out_path=str(out))
    samples, sr = sf.read(str(out), dtype="float32", always_2d=False)

    fmt = (job.get("settings") or {}).get("format", "wav")
    data, ext, mime = encode(samples, sr, fmt)
    path = f"{job['owner_id']}/{job['id']}/clean.{ext}"
    upload(sb, BUCKET_OUT, path, data, mime)

    # Level change, not a noise measurement — the honest name for what this is.
    # A real SNR would need the clean signal, which is the thing being produced.
    removed = round(dbfs(before) - dbfs(samples), 1)

    set_job(sb, job["id"], status="done", audio_path=path,
            duration_ms=int(len(samples) / sr * 1000), error=None,
            qc_summary={"noise_removed_db": removed, "sample_rate": int(sr),
                        **_model_info})
    _stats["audio_s"] += len(samples) / sr
    print(f"  cleaned — level down {removed} dB", flush=True)


def speech_job(sb, job):
    """Render a speak or batch job, line by line, resuming what is already done."""
    settings = job.get("settings") or {}
    gen = model_kwargs(settings)
    gap_ms = int(settings.get("join_gap_ms", 220))
    fmt = settings.get("format", "wav")
    kind = job.get("kind", "speak")

    work = scratch("vieneu-job-")
    model = get_model()

    lines = (sb.table("vieneu_lines").select("*").eq("job_id", job["id"])
             .order("idx").execute().data or [])
    if not lines:
        raise RuntimeError("This job has no lines to render.")

    job_voice = voice_for(sb, model, job.get("voice_preset"),
                          job.get("voice_id"), work)

    # Already-finished clips are kept and reused. Re-rolling one line of a
    # two-hundred-line batch has to cost one line, not two hundred — the same
    # contract the dub pipeline's per-cue re-roll gives.
    todo = [l for l in lines
            if l["status"] in ("pending", "rerolled", "failed") or not l.get("audio_path")]
    done_already = len(lines) - len(todo)

    print(f"  {len(lines)} line(s), {len(todo)} to render "
          f"({done_already} reused)", flush=True)
    set_job(sb, job["id"], total_lines=len(lines), done_lines=done_already)

    clips: dict[int, np.ndarray] = {}
    state = {"sr": 48_000, "done": done_already, "failed": 0}

    def keep(line, samples, wav: bytes):
        """Persist one finished line. Shared by the batched and per-line paths."""
        sr = state["sr"]
        path = f"{job['owner_id']}/{job['id']}/{line['idx']:04d}.wav"
        upload(sb, BUCKET_OUT, path, wav, "audio/wav")
        clips[line["idx"]] = samples
        sb.table("vieneu_lines").update({
            "status": "done", "audio_path": path, "error": None,
            "duration_ms": int(len(samples) / sr * 1000),
        }).eq("id", line["id"]).execute()
        _stats["lines"] += 1
        _stats["audio_s"] += len(samples) / sr
        state["done"] += 1
        set_progress(sb, job["id"], done_lines=state["done"])

    def blame(line, e):
        """One bad line must not cost the other hundred and ninety-nine."""
        state["failed"] += 1
        print(f"  line {line['idx']} failed: {e}", flush=True)
        sb.table("vieneu_lines").update({
            "status": "failed", "error": str(e)[:400],
        }).eq("id", line["id"]).execute()

    def one(line, voice):
        wav, samples, sr = render(model, line["body"], voice, work,
                                  f"line-{line['idx']:04d}", gen)
        state["sr"] = sr
        keep(line, samples, wav)

    # Lines are grouped by voice whatever the backend, because a group shares one
    # reference encode — and because `infer_batch` takes a single voice, so a
    # batch can never straddle two.
    groups: dict[str, list] = {}
    for line in todo:
        try:
            voice = (voice_for(sb, model, line.get("voice_preset"),
                               line.get("voice_id"), work)
                     if (line.get("voice_preset") or line.get("voice_id"))
                     else job_voice)
        except Exception as e:
            blame(line, e)          # a deleted or unconsented per-line voice
            continue
        groups.setdefault(voice, []).append(line)

    batch_size = settings.get("batch_size")
    # Batching is a GPU feature: on ONNX/CPU `infer_batch` runs the texts
    # sequentially anyway, and going through it there would only cost the
    # per-line error isolation for nothing.
    use_batch = (_model_info.get("backend") == "pytorch"
                 and batch_size != 1 and len(todo) > 1)

    n = 0
    for voice, group in groups.items():
        if use_batch and len(group) > 1:
            touch(sb, job, f"batch of {len(group)} — {job['title']}")
            try:
                kwargs = {"voice": voice, **gen}
                if batch_size:
                    kwargs["batch_size"] = int(batch_size)
                audios = model.infer_batch([l["body"] for l in group], **kwargs)
                for line, audio in zip(group, audios):
                    path = work / f"line-{line['idx']:04d}.wav"
                    model.save(audio, str(path))
                    samples, sr = sf.read(str(path), dtype="float32",
                                          always_2d=False)
                    state["sr"] = sr
                    keep(line, samples, path.read_bytes())
                n += len(group)
                continue
            except Exception as e:
                # A batch fails as a unit, so one unspeakable line would take
                # the whole group with it. Falling back line by line turns that
                # into one failed line, and costs only the batch already lost.
                print(f"  (batch of {len(group)} failed, retrying line by line: {e})",
                      flush=True)

        for line in group:
            # A batch that raised part way through has already banked the lines
            # it finished. Rendering them again would double the progress count
            # and pay twice for audio that is on disk.
            if line["idx"] in clips:
                continue
            n += 1
            touch(sb, job, f"line {n}/{len(todo)} — {job['title']}")
            try:
                one(line, voice)
            except Exception as e:
                blame(line, e)

    failures = state["failed"]

    # Reused clips have to come back from storage to be joined or zipped. Only
    # the ones that were not rendered in this pass — those are already in hand.
    reused = [l for l in lines if l["idx"] not in clips and l.get("audio_path")]
    for line in reused:
        try:
            blob = download(sb, BUCKET_OUT, line["audio_path"])
            tmp = work / f"reuse-{line['idx']:04d}.wav"
            tmp.write_bytes(blob)
            clips[line["idx"]], state["sr"] = sf.read(str(tmp), dtype="float32",
                                                       always_2d=False)
        except Exception as e:
            print(f"  (could not re-read line {line['idx']}: {e})", flush=True)

    sr = state["sr"]

    ordered = [clips[i] for i in sorted(clips)]
    if not ordered:
        raise RuntimeError("Every line failed — nothing to assemble.")

    joined = join(ordered, sr, gap_ms)
    data, ext, mime = encode(joined, sr, fmt)
    audio_path = f"{job['owner_id']}/{job['id']}/{'read' if kind == 'speak' else 'joined'}.{ext}"
    upload(sb, BUCKET_OUT, audio_path, data, mime)

    zip_path = None
    if kind == "batch":
        # A batch exists because the lines go to different places, so the whole
        # set as one download is the deliverable. The joined file above is a
        # convenience for listening through them, not the product.
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for line in lines:
                if line["idx"] not in clips:
                    continue
                one, _, _ = encode(clips[line["idx"]], sr, fmt)
                z.writestr(f"{line['idx'] + 1:04d}.{ext}", one)
        zip_path = f"{job['owner_id']}/{job['id']}/lines.zip"
        upload(sb, BUCKET_OUT, zip_path, buf.getvalue(), "application/zip")

    seconds = len(joined) / sr
    summary = {
        "voice": job.get("voice_preset") or "cloned",
        "chunks": len(lines),
        "rendered": len(clips),
        "failed": failures,
        "sample_rate": int(sr),
        **_model_info,
    }

    fields = {
        "status": "done", "audio_path": audio_path, "zip_path": zip_path,
        "duration_ms": int(seconds * 1000), "done_lines": len(clips),
        "total_lines": len(lines), "qc_summary": summary,
        "error": (f"{failures} line(s) failed — see the lines below"
                  if failures else None),
    }
    set_job(sb, job["id"], **fields)
    print(f"  done — {seconds:.1f}s audio"
          + (f", {failures} line(s) failed" if failures else ""), flush=True)


def run_job(sb, job):
    started = time.time()
    if job.get("kind") == "denoise":
        denoise_job(sb, job)
    else:
        speech_job(sb, job)

    # Recorded after the fact rather than measured per line: what anyone wants
    # to know is whether the machine keeps up with playback, and that includes
    # the uploads and the assembly, not just the model.
    try:
        rows = (sb.table("vieneu_jobs").select("duration_ms, qc_summary")
                .eq("id", job["id"]).limit(1).execute().data)
        if rows and rows[0].get("duration_ms"):
            rtf = (rows[0]["duration_ms"] / 1000) / max(time.time() - started, 0.01)
            summary = dict(rows[0].get("qc_summary") or {})
            summary["realtime_factor"] = round(rtf, 2)
            set_job(sb, job["id"], qc_summary=summary)
    except Exception:
        pass                       # a statistic, never worth failing a job for


# -------------------------------------------------------------------- loop

def claim_next_job(sb):
    """Atomically take one queued job. The status filter in the UPDATE is what
    stops two workers grabbing the same row."""
    rows = (sb.table("vieneu_jobs").select("*").eq("status", "queued")
            .order("priority").order("created_at").limit(1).execute().data)
    if not rows:
        return None
    job = rows[0]
    got = (sb.table("vieneu_jobs")
           .update({"status": "running", "claimed_at": now_iso(),
                    "heartbeat_at": now_iso()})
           .eq("id", job["id"]).eq("status", "queued").execute().data)
    return job if got else None


def claim_next_voice(sb):
    rows = (sb.table("vieneu_voices").select("*").eq("status", "uploaded")
            .order("created_at").limit(1).execute().data)
    if not rows:
        return None
    v = rows[0]
    got = (sb.table("vieneu_voices").update({"status": "building"})
           .eq("id", v["id"]).eq("status", "uploaded").execute().data)
    return v if got else None


def print_stats(t0):
    mins = (time.time() - t0) / 60
    rate = _stats["audio_s"] / max(time.time() - t0, 1)
    print(f"\n--- {mins:.1f} min | {_stats['jobs']} jobs, {_stats['voices']} voices, "
          f"{_stats['lines']} lines, {_stats['audio_s'] / 60:.1f} min audio "
          f"({rate:.2f}x realtime) ---", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true",
                    help="drain both queues and exit")
    ap.add_argument("--runtime-minutes", type=float,
                    default=float(os.environ.get("VIENEU_RUNTIME_MINUTES", "0")),
                    help="stop claiming new work after this long (0 = unlimited)")
    ap.add_argument("--requeue-stalled", action="store_true",
                    help="return jobs abandoned by a dead worker to the queue first")
    ap.add_argument("--list-voices", action="store_true",
                    help="print the model's preset voices and exit")
    args = ap.parse_args()

    if args.list_voices:
        model = get_model()
        for label, vid in model.list_preset_voices():
            print(f"  {label:<16} {vid}")
        return

    sb = get_client()
    t0 = time.time()
    budget = args.runtime_minutes * 60 if args.runtime_minutes > 0 else None

    if args.requeue_stalled:
        try:
            n = sb.rpc("requeue_stalled_vieneu", {"stale_minutes": 10}).execute().data
            print(f"requeued {n} stalled job(s)", flush=True)
        except Exception as e:
            print(f"(requeue_stalled_vieneu unavailable: {e})", flush=True)

    try:
        sb.rpc("prune_vieneu_workers", {"older_than_hours": 72}).execute()
    except Exception:
        pass                       # optional housekeeping, never worth failing

    beat(sb, "starting", detail="loading the model", force=True)
    print(f"vieneu worker up — {BACKEND}/{PRECISION}"
          + (f", budget {args.runtime_minutes:.0f} min" if budget else "")
          + f" — shows in the web app as '{worker_label()}'"
          + " (Ctrl+C to stop)", flush=True)

    idle = 0
    try:
        while True:
            if budget and time.time() - t0 > budget:
                print("\nruntime budget reached — stopping cleanly", flush=True)
                break
            refresh_auth(sb)

            voice = claim_next_voice(sb)
            if voice is not None:
                idle = 0
                beat(sb, "busy", detail=f"cloning {voice['name']}", force=True)
                try:
                    build_voice(sb, voice)
                except Exception as e:
                    traceback.print_exc()
                    sb.table("vieneu_voices").update(
                        {"status": "failed", "error": str(e)[:500]}
                    ).eq("id", voice["id"]).execute()
                _sweep_scratch()
                continue

            job = claim_next_job(sb)
            if job is not None:
                idle = 0
                print(f"\n=== {job['kind']} {job['id']} — {job['title']} ===",
                      flush=True)
                beat(sb, "busy", job_id=job["id"], detail=job["title"], force=True)
                jt = time.time()
                try:
                    run_job(sb, job)
                    _stats["jobs"] += 1
                    print(f"=== done in {time.time() - jt:.0f}s ===", flush=True)
                except Exception as e:
                    traceback.print_exc()
                    set_job(sb, job["id"], status="failed", error=str(e)[:500])
                print_stats(t0)
                _sweep_scratch()
                continue

            if args.once:
                print("both queues empty — exiting")
                break
            idle += 1
            if idle % 12 == 1:
                print("  ...idle", flush=True)
            beat(sb, "idle", detail="waiting for work")
            time.sleep(POLL)
    except KeyboardInterrupt:
        print("\nstopped by hand", flush=True)
    finally:
        _sweep_scratch()
        # Say goodbye rather than going stale, so the pill in the web app flips
        # the moment this exits instead of a minute later.
        beat(sb, "stopped", detail=None, force=True)

    print_stats(t0)


if __name__ == "__main__":
    main()
