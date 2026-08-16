"""GPU worker for the Dub Factory frontend.

Two queues, drained in one loop:
  1. voices with status='uploaded'  -> QC, transcribe, encode, audition, cache
  2. jobs   with status='queued'    -> render, verify, assemble, upload

Progress is written back to the tables the UI already subscribes to, so the
frontend updates live with no changes.

Why the worker pulls instead of the browser calling a GPU URL: the GPU is
ephemeral (Colab session, spot pod) while the database is not. A pull-based
worker survives restarts, needs no public URL or CORS, and several GPUs can
share one queue.

Credentials — one of these two sets:

    A. Self-hosted / own Supabase project
        SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

    B. Lovable Cloud (no service_role key is issued)
        SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, WORKER_EMAIL, WORKER_PASSWORD
       The worker signs in as a staff account and works under RLS. It therefore
       only sees jobs owned by that account — queue from the same login.

Other environment:
    OMNIVOICE_DEVICE             cuda:0
    WORKER_BATCH                 cues generated at once (default 4; raise on big GPUs)
    WORKER_ASR_BATCH             clips transcribed at once (default = WORKER_BATCH)
    WORKER_RUNTIME_MINUTES       stop claiming new work after N minutes (Colab budget)
    WORKER_POLL_SECONDS          idle poll interval (default 5)
    WORKER_ID                    presence row id (default: hostname-pid)
    WORKER_LABEL                 name shown in the web app (default: Colab / RunPod / hostname)

Run:
    python worker.py                       # loop until stopped
    python worker.py --once                # drain both queues then exit
    python worker.py --runtime-minutes 200 # stop cleanly before a Colab session ends
"""

import argparse
import io
import os
import platform
import re
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

import speakers as SPK
import srt_dub as S
import voice_lib as VL

SR = S.SR
POLL = float(os.environ.get("WORKER_POLL_SECONDS", "5"))
DEVICE = os.environ.get("OMNIVOICE_DEVICE", "cuda:0")
BATCH = int(os.environ.get("WORKER_BATCH", "4"))
ASR_BATCH = int(os.environ.get("WORKER_ASR_BATCH", str(BATCH)))

# On-screen labels rather than speech: "(Yoon Hyun Woo)", "[phone rings]".
CAPTION_RE = re.compile(r"^\s*[\(\[\{].*[\)\]\}]\s*$", re.DOTALL)

_model = None
_prompts: dict[str, object] = {}
_timbres: dict[str, np.ndarray] = {}
_stats = {"jobs": 0, "voices": 0, "cues": 0, "audio_s": 0.0, "gpu_s": 0.0}


# ----------------------------------------------------------------- supabase

_auth = {"email": None, "password": None, "at": 0.0, "uid": None}


def get_client():
    """Connect either as service_role, or as a signed-in user.

    Lovable Cloud manages its Supabase project and does not hand out the
    service_role key, so the user path is the one that works there. The RLS
    policies grant `authenticated` full CRUD on rows where owner_id = auth.uid(),
    which is exactly what a worker needs — it simply only sees that account's
    jobs, rather than everyone's.
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
    _auth.update(email=email, password=password, at=time.time(), uid=res.user.id)
    print(f"auth: signed in as {email} (only this account's jobs are visible)",
          flush=True)
    return sb


# Supabase access tokens last an hour. Refreshing at half that leaves room for
# one failed attempt and a slow retry before anything actually expires.
TOKEN_MAX_AGE = 30 * 60


def refresh_auth(sb, force: bool = False):
    """Re-sign-in before the JWT expires.

    Must be called from inside a render, not only between jobs. A 400-cue
    episode takes about an hour, which is exactly the token's lifetime — one
    was lost after all 411 cues had rendered, three minutes into uploading the
    result, to `"exp" claim timestamp check failed`. Every second of that GPU
    time was already spent.

    Cheap to call often: it returns immediately unless the token is old.
    """
    if not _auth["email"]:
        return                      # service_role key — never expires
    age = time.time() - _auth["at"]
    if not force and age < TOKEN_MAX_AGE:
        return
    try:
        sb.auth.sign_in_with_password({"email": _auth["email"],
                                       "password": _auth["password"]})
        _auth["at"] = time.time()
        print(f"  (auth token refreshed after {age / 60:.0f} min)", flush=True)
    except Exception as e:
        # Leave _auth["at"] alone so the next call tries again rather than
        # waiting another full interval on a token that is already stale.
        print(f"  (token refresh failed, will retry: {e})", flush=True)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def log(sb, job, stage, message, details=None):
    print(f"  [{stage}] {message}", flush=True)
    try:
        sb.table("job_events").insert({
            "owner_id": job["owner_id"], "job_id": job["id"],
            "stage": stage, "message": message, "details": details,
        }).execute()
    except Exception as e:                       # logging must never kill a job
        print(f"  (job_events write failed: {e})", flush=True)


def set_job(sb, job_id, **fields):
    sb.table("jobs").update(fields).eq("id", job_id).execute()


def set_job_progress(sb, job_id, **fields):
    """Progress writes must never kill a render.

    The clips this update reports on are already uploaded; losing one counter
    refresh to a network blip is nothing, losing the episode to it is hours.
    """
    try:
        set_job(sb, job_id, **fields)
    except Exception as e:
        print(f"  (progress update failed: {e})", flush=True)


def upload_output(sb, path, data: bytes, content_type: str):
    """Upload to the outputs bucket, retrying once on a fresh token.

    The final uploads run at the exact moment the access token is oldest —
    after the whole episode has rendered — which is also the most expensive
    possible moment to fail. One refresh-and-retry covers both an expired
    token and a transient blip; a second failure is a real outage and raises.
    """
    try:
        sb.storage.from_("outputs").upload(
            path, data, {"content-type": content_type, "upsert": "true"})
    except Exception as e:
        print(f"  (upload {path} failed, retrying once: {e})", flush=True)
        refresh_auth(sb, force=True)
        sb.storage.from_("outputs").upload(
            path, data, {"content-type": content_type, "upsert": "true"})


def has_column(sb, table, column):
    """Columns from the voice-library migration are optional; degrade politely."""
    key = f"{table}.{column}"
    if key not in _COLS:
        try:
            sb.table(table).select(column).limit(1).execute()
            _COLS[key] = True
        except Exception:
            _COLS[key] = False
            print(f"  (note: {key} missing — run the voice_library migration "
                  f"to enable that feature)", flush=True)
    return _COLS[key]


_COLS: dict[str, bool] = {}


# ----------------------------------------------------------------- presence

# The web app answers "is the render engine on?" from these rows. It has to keep
# ticking while the queue is empty too: a worker waiting for work is still a GPU
# that is up and will pick the next job straight away, which is exactly what
# somebody about to press "Queue render" wants to know.
BEAT_SECONDS = 20

WORKER_ID = os.environ.get("WORKER_ID") or f"{platform.node() or 'worker'}-{os.getpid()}"
_beat = {"at": 0.0, "gpu": None, "probed": False, "on": True}


def worker_label():
    if os.environ.get("WORKER_LABEL"):
        return os.environ["WORKER_LABEL"]
    if os.environ.get("COLAB_RELEASE_TAG") or os.path.isdir("/content/drive"):
        return "Colab"
    if os.environ.get("RUNPOD_POD_ID"):
        return f"RunPod {os.environ['RUNPOD_POD_ID'][:8]}"
    return platform.node() or "worker"


def gpu_name():
    """Probe once. torch is already a dependency, so this costs nothing after
    the first call and gives the UI something better than 'a worker'."""
    if not _beat["probed"]:
        _beat["probed"] = True
        try:
            import torch
            if torch.cuda.is_available():
                _beat["gpu"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return _beat["gpu"]


def beat(sb, status, job_id=None, detail=None, force=False):
    """Write this worker's presence row. Rate-limited unless forced.

    Presence must never be the thing that breaks a render, and the table is
    optional — a project that has not run the engine-status migration keeps
    working, it just cannot show the indicator. So one failure turns it off for
    the rest of the session rather than raising on every loop.
    """
    if not _beat["on"]:
        return
    if not force and time.time() - _beat["at"] < BEAT_SECONDS:
        return
    try:
        sb.table("render_workers").upsert({
            "id": WORKER_ID, "label": worker_label(), "gpu": gpu_name(),
            "status": status, "job_id": job_id, "detail": detail,
            "last_seen_at": now_iso(),
            "stats": {"jobs": _stats["jobs"], "voices": _stats["voices"],
                      "cues": _stats["cues"],
                      "audio_min": round(_stats["audio_s"] / 60, 1)},
        }).execute()
        _beat["at"] = time.time()
    except Exception as e:
        _beat["on"] = False
        print(f"  (engine status off — render_workers unavailable: {e})", flush=True)


_touch = {"at": 0.0}


def touch_job(sb, job, detail=None):
    """Prove a job is alive during its long non-rendering stages.

    The stall sweep requeues any job whose heartbeat is older than 15 minutes.
    Rendering heartbeats every batch, but adaptation (minutes of LLM calls)
    and muxing (Demucs on a full episode) used to go silent for their whole
    duration — long enough to be mistaken for a dead worker and handed to a
    second one, which would then render the same episode twice.

    Cheap to call from any progress callback: rate-limited to one write per
    30 s, renews the auth token on the way, and never raises — a failed
    heartbeat must not kill the work it exists to protect.
    """
    if time.time() - _touch["at"] < 30:
        return
    _touch["at"] = time.time()
    try:
        refresh_auth(sb)
        if has_column(sb, "jobs", "heartbeat_at"):
            set_job(sb, job["id"], heartbeat_at=now_iso())
        beat(sb, "busy", job_id=job["id"], detail=detail or job.get("title"))
    except Exception as e:
        print(f"  (heartbeat failed: {e})", flush=True)


# -------------------------------------------------------------------- model

def get_model():
    global _model
    if _model is None:
        import torch
        from omnivoice import OmniVoice
        print(f"loading OmniVoice on {DEVICE} ...", flush=True)
        _model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice", device_map=DEVICE, dtype=torch.float16
        )
        _model.load_asr_model()
        print(f"model + ASR ready (gen batch {BATCH}, asr batch {ASR_BATCH})", flush=True)
    return _model


# Whisper's feature extractor cannot build a spectrogram from silence shorter
# than one frame; handed a zero-length array it raises
#   cannot reshape tensor of 0 elements into shape [-1, 0]
# which used to escape render_job and fail the whole episode. Generation does
# occasionally return no samples — a cue whose text has nothing pronounceable in
# it ("...") reproduces it every time — so this has to be handled, not avoided.
MIN_ASR_SAMPLES = SR // 20        # 50 ms; below this there is nothing to hear


def transcribe_many(model, clips: list[np.ndarray]) -> list[str]:
    """Transcribe a list of clips, batched through the underlying HF pipeline.

    model.transcribe() handles one clip per call; the pipeline it wraps accepts
    a list, which keeps the GPU busy instead of paying per-call overhead on
    every short cue. Falls back to one-at-a-time if anything about the batched
    path fails, since verification must never be the thing that breaks a job.

    Clips too short to transcribe come back as "" rather than raising. That is
    the honest answer — nothing was said — and it flows through coverage() as
    0% spoken, so the cue is retried like any other bad take instead of taking
    the episode down with it.
    """
    if not clips:
        return []

    sayable = [i for i, c in enumerate(clips) if c.size >= MIN_ASR_SAMPLES]
    texts = [""] * len(clips)
    if len(sayable) != len(clips):
        print(f"  ({len(clips) - len(sayable)} clip(s) too short to transcribe "
              f"— counted as nothing spoken)", flush=True)
    if not sayable:
        return texts

    batch = [clips[i].astype(np.float32) for i in sayable]
    try:
        inputs = [{"array": c, "sampling_rate": SR} for c in batch]
        out = model._asr_pipe(inputs, batch_size=max(1, ASR_BATCH))
        if isinstance(out, dict):
            out = [out]
        for i, o in zip(sayable, out):
            texts[i] = str(o["text"]).strip()
        return texts
    except Exception as e:
        print(f"  (batched ASR unavailable, falling back: {e})", flush=True)

    import torch
    for i in sayable:
        # One clip failing to transcribe must not lose the verification for the
        # rest of the batch — an empty transcript just means "retry this cue".
        try:
            t = model.transcribe((torch.from_numpy(clips[i]), SR))
            texts[i] = str(t[0] if isinstance(t, (list, tuple)) else t).strip()
        except Exception as e:
            print(f"  (clip {i} could not be transcribed: {e})", flush=True)
    return texts


# ------------------------------------------------------------ voice library

def process_voice(sb, voice):
    """QC → transcribe → encode → audition → cache. Runs once per uploaded voice."""
    vid = voice["id"]
    work = Path(tempfile.mkdtemp(prefix=f"voice_{vid[:8]}_"))
    print(f"\n=== voice {vid} — {voice['name']} ===", flush=True)
    sb.table("voices").update({"status": "encoding", "error": None}).eq("id", vid).execute()

    model = get_model()
    language = voice.get("language_name") or "Vietnamese"
    designed = (voice.get("kind") or "cloned") == "designed"

    if designed:
        if not voice.get("instruct"):
            raise RuntimeError("designed voice has no attribute description")
        print(f"  designing from: {voice['instruct']}", flush=True)
        built = VL.build_voice(model, work, language=language,
                               instruct=voice["instruct"])
    else:
        if not voice.get("sample_path"):
            raise RuntimeError("voice has no sample uploaded")
        local = work / "sample.wav"
        local.write_bytes(sb.storage.from_("voices").download(voice["sample_path"]))
        built = VL.build_voice(model, work, language=language,
                               sample_path=str(local),
                               ref_text=(voice.get("ref_text") or None))

    update = {"status": "ready", "error": None}
    base = f"{voice['owner_id']}/{vid}"

    # A designed voice has no upload, so its seed clip becomes the sample —
    # that is the recording the voice was actually frozen from.
    if designed and built.get("seed_file") is not None:
        p = f"{base}/designed_seed.wav"
        sb.storage.from_("voices").upload(
            p, built["seed_file"].read_bytes(),
            {"content-type": "audio/wav", "upsert": "true"})
        update["sample_path"] = p

    if has_column(sb, "voices", "prompt_path"):
        p = f"{base}/prompt.pt"
        sb.storage.from_("voices").upload(
            p, built["prompt_file"].read_bytes(),
            {"content-type": "application/octet-stream", "upsert": "true"})
        update["prompt_path"] = p
    if has_column(sb, "voices", "preview_path"):
        p = f"{base}/preview.wav"
        sb.storage.from_("voices").upload(
            p, built["preview_file"].read_bytes(),
            {"content-type": "audio/wav", "upsert": "true"})
        update["preview_path"] = p
    if has_column(sb, "voices", "ref_text"):
        update["ref_text"] = built["ref_text"]
    if has_column(sb, "voices", "qc"):
        update["qc"] = built["qc"]
    if has_column(sb, "voices", "qc_passed"):
        update["qc_passed"] = built["qc_passed"]

    sb.table("voices").update(update).eq("id", vid).execute()
    _stats["voices"] += 1
    q = built["qc"]
    print(f"  ready — {q['duration_s']}s, SNR {q.get('snr_db')} dB, "
          f"self-similarity {q.get('self_similarity')}", flush=True)
    for w in q["warnings"]:
        print(f"  ! {w}", flush=True)


def get_voice_prompt(sb, vrow, workdir: Path):
    """Load a voice's clone prompt — from memory, then storage cache, then encode."""
    vid = vrow["id"]
    if vid in _prompts:
        return _prompts[vid], _timbres[vid]

    model = get_model()
    sample = None
    if vrow.get("sample_path"):
        sample = workdir / f"sample_{vid}.wav"
        sample.write_bytes(sb.storage.from_("voices").download(vrow["sample_path"]))

    prompt = None
    if vrow.get("prompt_path"):
        try:
            blob = sb.storage.from_("voices").download(vrow["prompt_path"])
            prompt = VL.load_cached_prompt(model, blob, workdir)
            print("  voice prompt loaded from cache", flush=True)
        except Exception as e:
            print(f"  (cached prompt unusable, re-encoding: {e})", flush=True)

    if prompt is None:
        if sample is None:
            raise RuntimeError("voice has neither a cached prompt nor a sample")
        prompt = model.create_voice_clone_prompt(
            str(sample), ref_text=(vrow.get("ref_text") or None))

    if sample is not None:
        wav, _ = sf.read(sample, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        timbre = S.timbre_vector(wav)
    else:
        timbre = np.zeros(20, dtype=np.float32)

    _prompts[vid], _timbres[vid] = prompt, timbre
    return prompt, timbre


# ------------------------------------------------------------------ render

def cue_text(c) -> str:
    """What to speak for a cue.

    Two layers sit between the uploaded line and the spoken one, and both are
    resolved here so that generation, ASR verification and the emitted .srt can
    never disagree about what was said:

      * adaptation writes its rewrite to final_text, leaving source_text as the
        untouched record of what was uploaded;
      * casting stores the character in `speaker` and leaves its label in the
        text, so the label is removed at render time rather than at upload —
        clearing a wrongly-detected speaker restores the whole line.
    """
    text = (c.get("final_text") or "").strip() or c["source_text"]
    return SPK.strip_label(text, c.get("speaker"))


def resolve_voice(c, default_voice_id):
    """Which voice speaks this cue — its cast voice, else the job's."""
    return c.get("voice_id") or default_voice_id


def adapt_cues(sb, job, cues, language, settings):
    """Rewrite over-long lines to fit their slots, before any GPU time is spent.

    Only cues without a final_text are considered, so re-rendering a job does
    not pay for the same rewrites twice — and a cue whose text was edited in
    the UI (which clears final_text) is re-adapted with its new wording.
    """
    import adapt_srt as A

    pending = [c for c in cues if not (c.get("final_text") or "").strip()]
    if not pending:
        return
    # Budget the SPOKEN text. source_text still carries the speaker label
    # ("MINH: …"), which is stripped at render time — counting it here inflates
    # every labeled line by its name's syllables, and sending it to the model
    # asks it to rewrite a label nobody will speak.
    probes = [S.Cue(c["idx"], c["start_ms"] / 1000.0, c["end_ms"] / 1000.0,
                    SPK.strip_label(c["source_text"], c.get("speaker")))
              for c in pending]
    syl = float(settings.get("syllables_per_sec", S.SYLLABLES_PER_SEC))
    tol = float(settings.get("adapt_tolerance", A.DEFAULT_TOLERANCE))
    rows, targets = A.analyse(probes, syl, tol)
    over = sum(1 for r in rows if r["over"])
    if not targets:
        log(sb, job, "adapting",
            f"All {len(probes)} line(s) already fit their slots — nothing to rewrite")
        return

    log(sb, job, "adapting",
        f"{len(targets)} of {len(probes)} line(s) need shortening or "
        f"number-spelling ({over} over budget)")
    def _adapt_log(m):
        print(f"  [adapting] {m}", flush=True)
        touch_job(sb, job, detail="adapting over-long lines")

    try:
        new_text = A.adapt(
            probes, targets, language=language,
            provider=settings.get("adapt_provider", "auto"),
            model=settings.get("adapt_model"),
            syllables_per_sec=syl,
            log=_adapt_log,
            tick=lambda: touch_job(sb, job, detail="adapting over-long lines"))
    except (A.NoCredentials, A.BadModel) as e:
        # Never fail a render over this: the original text still speaks, it
        # just may overrun. Surface it so staff know why nothing was rewritten.
        log(sb, job, "adapting", f"Skipped — {str(e).splitlines()[0]}")
        return
    except Exception as e:
        log(sb, job, "adapting", f"Skipped after an error: {str(e)[:200]}")
        return

    by_idx = {c["idx"]: c for c in pending}
    for idx, text in new_text.items():
        row = by_idx.get(idx)
        if row is not None:
            sb.table("cues").update({"final_text": text}).eq("id", row["id"]).execute()
            row["final_text"] = text

    after, _ = A.analyse(
        [S.Cue(c["idx"], c["start_ms"] / 1000.0, c["end_ms"] / 1000.0, cue_text(c))
         for c in pending], syl, tol)
    summary = {
        "lines_considered": len(probes),
        "lines_over_budget": over,
        "lines_rewritten": len(new_text),
        "worst_ratio_before": round(max((r["ratio"] for r in rows), default=0), 2),
        "worst_ratio_after": round(max((r["ratio"] for r in after), default=0), 2),
        "still_over_budget": sum(1 for r in after if r["over"]),
        "language": language,
    }
    log(sb, job, "adapting",
        f"Rewrote {len(new_text)} line(s); worst fit "
        f"{summary['worst_ratio_before']}x -> {summary['worst_ratio_after']}x",
        summary)
    if has_column(sb, "jobs", "adapt_summary"):
        set_job(sb, job["id"], adapt_summary=summary)


def mux_video(sb, job, work, dub_wav):
    """Lay the finished dub over the source video, keeping music and effects.

    Returns the storage path of the dubbed MP4, or None when the job has no
    video or the mux could not run. A failure here never fails the job — the
    dub.wav is already uploaded and usable on its own.
    """
    video_path = job.get("video_path")
    if not video_path:
        return None
    if not shutil.which("ffmpeg"):
        log(sb, job, "muxing", "Skipped — ffmpeg is not installed on this worker")
        return None

    import video_dub as V

    job_id, owner = job["id"], job["owner_id"]
    settings = job.get("settings") or {}
    set_job(sb, job_id, status="muxing")
    log(sb, job, "muxing", "Building the dubbed video")

    local_video = work / ("source" + Path(video_path).suffix.lower())
    local_video.write_bytes(sb.storage.from_("videos").download(video_path))

    def _mux_log(m):
        print(f"  [muxing] {m}", flush=True)
        touch_job(sb, job, detail="mixing the dubbed video")

    out_mp4 = work / "dubbed.mp4"
    info = V.mux_dub(
        str(local_video), str(dub_wav), str(out_mp4),
        duck_db=settings.get("duck_db"),
        # An absent key means the default blend; an explicit null means the job
        # asked for a pure separated bed, so .get's default cannot be reused.
        residual_db=settings.get("residual_db", V.RESIDUAL_DB),
        separate=bool(settings.get("separate_voices", True)),
        log=_mux_log,
        # Demucs on a full episode runs for many silent minutes; the tick is
        # what keeps the stall sweep from handing the job to a second worker.
        tick=lambda: touch_job(sb, job, detail="separating original voices"),
    )

    mp4_path = f"{owner}/{job_id}/dubbed.mp4"
    upload_output(sb, mp4_path, out_mp4.read_bytes(), "video/mp4")
    blend = info.get("residual_db")
    if not info["separated"]:
        note = ("Original mix ducked under the dub (separation turned off for this job)"
                if not settings.get("separate_voices", True) else
                "Original mix ducked under the dub — install Demucs on the "
                "worker to remove the original voices")
    elif not info.get("silent_bed"):
        note = "Music and effects kept, original voices removed"
    elif blend is None:
        # The one outcome that sounds broken rather than merely imperfect, so
        # it is named on the job instead of left to be discovered on playback.
        note = ("This source has no music or effects that survive separation, "
                "and the ambience blend is off — the dub plays over near-"
                "silence. Set an ambience blend, or keep the original mix.")
    else:
        note = ("Original voices removed. This source had no separable music, "
                f"so the background is the ambience blend at {blend:.0f} dB")
    log(sb, job, "muxing", note, info)
    return mp4_path, info


def render_job(sb, job):
    job_id, owner = job["id"], job["owner_id"]
    settings = job.get("settings") or {}
    kind = job.get("kind") or "subtitles"     # subtitles | video | tts
    read_captions = bool(settings.get("read_caption_cues", False))
    timing_mode = settings.get("timing_mode", "natural")      # natural | fit
    language = settings.get("language", "Vietnamese")
    max_attempts = int(settings.get("max_attempts", 4))

    # A text-to-speech job is prose, not subtitles. Two subtitle conventions
    # actively damage it and are switched off:
    #   * caption skipping would silently drop a paragraph that happens to be
    #     wholly parenthesised — "(See appendix A.)" is content in a document;
    #   * fit-to-timecode has nothing to fit to, since the timestamps are
    #     estimates this pipeline generated rather than an edit to honour.
    if kind == "tts":
        read_captions = True
        timing_mode = "natural"

    work = Path(tempfile.mkdtemp(prefix=f"job_{job_id[:8]}_"))
    set_job(sb, job_id, status="compiling")
    log(sb, job, "compiling", "Reading cues")

    cues = sb.table("cues").select("*").eq("job_id", job_id).order("idx").execute().data
    if not cues:
        raise RuntimeError("job has no cues")
    if not job.get("voice_id"):
        raise RuntimeError("job has no voice selected")

    vrow = sb.table("voices").select("*").eq("id", job["voice_id"]).single().execute().data
    if not vrow.get("consent_confirmed"):
        raise RuntimeError("voice has no consent on record")

    speak, skipped = [], []
    for c in cues:
        (skipped if (not read_captions and CAPTION_RE.match(c["source_text"])) else speak).append(c)
    for c in skipped:
        sb.table("cues").update({"status": "condensed", "final_text": "",
                                 "note": "caption cue — not read"}).eq("id", c["id"]).execute()
    if skipped:
        log(sb, job, "compiling", f"Skipped {len(skipped)} caption cue(s)")

    # Adaptation runs before any GPU time is spent: a line rewritten to fit its
    # slot never has to be rescued later by dropping words or crushing audio.
    if settings.get("adapt_text"):
        set_job(sb, job_id, status="adapting")
        adapt_cues(sb, job, speak, language, settings)

    # A requeued job keeps its finished takes. Only re-rolled cues, cues that
    # never rendered, and cues whose stored clip cannot be fetched are
    # regenerated; on a first render every cue is pending with no audio, so
    # this partition degenerates to "render everything".
    todo, reuse = [], []
    for c in speak:
        if c.get("status") == "rerolled" or not c.get("audio_path"):
            todo.append(c)
        else:
            reuse.append(c)

    clips: dict[int, np.ndarray] = {}
    for c in list(reuse):
        try:
            blob = sb.storage.from_("outputs").download(c["audio_path"])
            wav, _ = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            clips[c["idx"]] = wav
        except Exception as e:
            print(f"  cue {c['idx']}: stored clip unreadable ({e}) — re-rendering",
                  flush=True)
            reuse.remove(c)
            todo.append(c)
    todo.sort(key=lambda x: x["idx"])

    # Casting: each character may have its own voice, falling back to the job's.
    # Cues are grouped by voice and rendered a group at a time — a batch shares
    # one clone prompt, so mixing voices inside a batch is not an option, and
    # grouping also means each voice is encoded once rather than per batch.
    by_voice: dict[str, list] = {}
    for c in todo:
        by_voice.setdefault(resolve_voice(c, job["voice_id"]), []).append(c)

    voices = {}
    if todo:
        model = get_model()
        for vid in by_voice:
            row = (vrow if vid == job["voice_id"] else
                   sb.table("voices").select("*").eq("id", vid).single().execute().data)
            if not row.get("consent_confirmed"):
                raise RuntimeError(
                    f"cast voice {row.get('name', vid)!r} has no consent on record")
            voices[vid] = (row, *get_voice_prompt(sb, row, work))

    done = len(skipped) + len(reuse)
    review = sum(1 for c in reuse if c.get("status") in ("review", "qc_fail"))
    set_job(sb, job_id, status="rendering", total_cues=len(cues), done_cues=done,
            review_cues=review)
    if reuse:
        log(sb, job, "rendering",
            f"Re-render: generating {len(todo)} cue(s), keeping "
            f"{len(reuse)} finished take(s)")
    elif len(by_voice) > 1:
        cast = ", ".join(f"{voices[v][0]['name']} ({len(cs)})"
                         for v, cs in by_voice.items())
        log(sb, job, "rendering",
            f"Rendering {len(todo)} cue(s) across {len(by_voice)} voices: {cast}")
    else:
        log(sb, job, "rendering", f"Rendering {len(todo)} cue(s) as {vrow['name']}")

    job_beat = time.time()

    # Batches are cut per voice group rather than across a flat list, so a batch
    # can never straddle two voices — slicing a flat list and trimming at the
    # boundary would silently skip whatever was trimmed off.
    batches = [(vid, group[i:i + BATCH])
               for vid, group in by_voice.items()
               for i in range(0, len(group), BATCH)]
    # Silently dropping cues is exactly the failure this grouping guards
    # against, so check it rather than assert it — asserts vanish under -O.
    planned = sum(len(b) for _, b in batches)
    if planned != len(todo):
        raise RuntimeError(
            f"batching lost cues: planned {planned} of {len(todo)}")

    failed_batches = 0
    for voice_id, batch in batches:
        # Renew the token before the batch, not after: this batch is about to
        # upload a dozen clips, and an episode outlives its access token.
        # Rate-limited internally, so calling it every batch costs nothing.
        refresh_auth(sb)

        _, prompt, ref_timbre = voices[voice_id]
        best = {c["id"]: None for c in batch}
        pending = list(batch)

        for attempt in range(1, max_attempts + 1):
            kw = {"language": language}
            if attempt > 1:
                kw["duration"] = [
                    S.natural_estimate(cue_text(c)) * (1 + 0.1 * (attempt - 1)) / S.OVERSHOOT
                    for c in pending
                ]
            # An episode is hours of GPU time. Losing all of it because one
            # batch raised is the expensive failure, so a batch that breaks is
            # abandoned and its cues flagged, while everything already rendered
            # is kept and the job goes on to finish.
            try:
                outs = model.generate(
                    text=[cue_text(c) for c in pending],
                    voice_clone_prompt=[prompt] * len(pending), **kw,
                )
                wavs = [np.asarray(a, dtype=np.float32).squeeze() for a in outs]
                heard = transcribe_many(model, wavs)
            except Exception as e:
                failed_batches += 1
                traceback.print_exc()
                log(sb, job, "rendering",
                    f"Cue(s) {[c['idx'] for c in pending]} could not be "
                    f"rendered and were skipped: {str(e)[:160]}")
                break

            still = []
            for c, wav, txt in zip(pending, wavs, heard):
                cov, missing, tonal = S.coverage(cue_text(c), txt)
                sim = S.cosine(ref_timbre, S.timbre_vector(wav))
                prev = best[c["id"]]
                if prev is None or (cov, sim) > (prev[1], prev[2]):
                    best[c["id"]] = (wav, cov, sim, missing, tonal)
                if cov < 1.0:
                    still.append(c)
            pending = still
            if not pending:
                break

        for c in batch:
            take = best[c["id"]]
            # No take at all — the batch broke before producing one. Flag the
            # cue so it shows up for re-roll and carry on with the episode.
            if take is None:
                review += 1
                done += 1
                sb.table("cues").update({
                    "status": "qc_fail", "cer": 1.0, "audio_path": None,
                    "rendered_ms": 0, "final_text": cue_text(c),
                    "note": "render failed — re-roll this line",
                }).eq("id", c["id"]).execute()
                continue

            wav, cov, sim, missing, tonal = take
            # A zero-length take is silence, not audio: keeping it would write an
            # empty clip and place a 0s hole in the timeline.
            if wav.size == 0:
                review += 1
                done += 1
                sb.table("cues").update({
                    "status": "qc_fail", "cer": 1.0, "audio_path": None,
                    "rendered_ms": 0, "final_text": cue_text(c),
                    "note": "model produced no audio — re-roll, or reword this line",
                }).eq("id", c["id"]).execute()
                continue

            clips[c["idx"]] = wav
            _stats["cues"] += 1
            _stats["audio_s"] += wav.size / SR
            cer = round(1.0 - cov, 4)                 # 0.0 = every word spoken
            if cov >= 1.0:
                status = "qc_pass"
                note = f"tone check: {', '.join(tonal)}" if tonal else None
            else:
                status = "review" if cov >= 0.9 else "qc_fail"
                note = f"missing: {', '.join(missing)}"
                review += 1
            path = f"{owner}/{job_id}/cue_{c['idx']:05d}.wav"
            local = work / f"cue_{c['idx']:05d}.wav"
            sf.write(local, wav, SR)
            try:
                upload_output(sb, path, local.read_bytes(), "audio/wav")
            except Exception as e:
                print(f"  cue {c['idx']} upload failed twice: {e}", flush=True)
                path = None
            done += 1
            sb.table("cues").update({
                "status": status, "cer": cer, "audio_path": path,
                "rendered_ms": int(wav.size / SR * 1000),
                "final_text": cue_text(c), "note": note,
            }).eq("id", c["id"]).execute()

        fields = {"done_cues": done, "review_cues": review}
        if time.time() - job_beat > 30 and has_column(sb, "jobs", "heartbeat_at"):
            fields["heartbeat_at"] = now_iso()
            job_beat = time.time()
        set_job_progress(sb, job_id, **fields)
        beat(sb, "busy", job_id=job_id, detail=f"cue {done}/{len(cues)} — {job['title']}")

    # ------------------------------------------------------------ assemble
    # Everything from here is uploads — the dub, the .srt, maybe a video — and
    # it is the most expensive possible moment to be told the token expired,
    # because the whole episode is already rendered. Start it on a fresh one.
    refresh_auth(sb, force=True)
    set_job(sb, job_id, status="qc")
    log(sb, job, "qc", f"{done - review}/{done} cues passed verification")
    set_job(sb, job_id, status="assembling")

    placed, cursor = [], 0.0
    for c in sorted(speak, key=lambda x: x["idx"]):
        clip = clips.get(c["idx"])
        if clip is None or clip.size == 0:
            continue
        # A subtitle cue starts at its timestamp, or later if the previous clip
        # is still running. A read-aloud has no timestamps worth honouring —
        # they were estimated from the text — so its clips simply follow one
        # another at the cursor and the paragraph rhythm comes from the writing.
        start_s = cursor if kind == "tts" else max(c["start_ms"] / 1000.0, cursor)
        cursor = start_s + clip.size / SR + S.GAP
        placed.append((c, clip, start_s))

    srt_end = max(c["end_ms"] for c in cues) / 1000.0
    natural_end = max((s + c.size / SR for _, c, s in placed), default=0.0)
    speed = 1.0
    if timing_mode == "fit" and natural_end > srt_end > 0:
        speed = natural_end / srt_end
        log(sb, job, "assembling", f"Fitting to timecode: {speed:.2f}x")

    end = natural_end / speed
    timeline = np.zeros(int(round(end * SR)) + SR, dtype=np.float32)
    final_times = []
    for c, clip, start_s in placed:
        if speed > 1.001:
            import librosa
            clip = librosa.effects.time_stretch(clip, rate=speed)
            start_s = start_s / speed
        pos = int(round(start_s * SR))
        seg = timeline[pos:pos + clip.size]
        seg += clip[:seg.size]
        final_times.append((c, start_s, start_s + clip.size / SR))
    peak = float(np.max(np.abs(timeline))) if timeline.size else 0.0
    if peak > 1.0:
        timeline /= peak

    out_wav = work / "dub.wav"
    sf.write(out_wav, timeline, SR)
    wav_path = f"{owner}/{job_id}/dub.wav"
    upload_output(sb, wav_path, out_wav.read_bytes(), "audio/wav")

    fields = {"status": "done", "wav_path": wav_path, "error": None,
              "done_cues": done, "review_cues": review}

    # Natural-mode audio runs longer than the source subtitles, so a corrected
    # .srt is what makes the result usable in an editor. It carries the spoken
    # wording, which after adaptation is not the uploaded wording.
    if has_column(sb, "jobs", "srt_out_path"):
        lines = []
        for n, (c, a, b) in enumerate(final_times, 1):
            lines.append(f"{n}\n{S.fmt_ts(a)} --> {S.fmt_ts(b)}\n{cue_text(c)}\n")
        srt_path = f"{owner}/{job_id}/dub.srt"
        upload_output(sb, srt_path, ("\n".join(lines)).encode("utf-8"),
                      "application/x-subrip")
        fields["srt_out_path"] = srt_path

    cer_rows = sb.table("cues").select("cer").eq("job_id", job_id).execute().data
    cers = [r["cer"] for r in cer_rows if r.get("cer") is not None]
    qc = {
        "word_coverage": round((1 - float(np.mean(cers))) * 100, 2) if cers else 0.0,
        "cues_total": len(cues), "cues_spoken": len(placed),
        "cues_skipped": len(skipped), "cues_needing_review": review,
        "timing_mode": timing_mode, "speed_applied": round(speed, 3),
        "audio_seconds": round(end, 2),
        "voice": vrow["name"], "kind": kind,
    }
    if failed_batches:
        # The episode finished, but not all of it rendered. Say so on the job
        # rather than letting a quietly shorter dub look like a clean run.
        qc["failed_batches"] = failed_batches
        log(sb, job, "qc",
            f"{failed_batches} batch(es) failed to render; those cues are "
            f"flagged for re-roll and are silent in the mix")
    # Only a job with a real timecode can overrun it. For a read-aloud the
    # "srt seconds" would just be this pipeline's own estimate, and reporting
    # the audio as running over its own guess is noise, not information.
    if kind != "tts":
        qc["srt_seconds"] = round(srt_end, 2)
        qc["overrun_seconds"] = round(max(0.0, end - srt_end), 2)

    # Casting is worth recording per job: "which voice played whom" is the first
    # thing anyone asks when a rendered episode sounds wrong.
    cast_rows = {}
    for c in speak:
        who = c.get("speaker")
        if who:
            cast_rows.setdefault(who, {"lines": 0, "voice": None})["lines"] += 1
            vid = resolve_voice(c, job["voice_id"])
            if vid in voices:
                cast_rows[who]["voice"] = voices[vid][0]["name"]
    if cast_rows:
        qc["cast"] = cast_rows
        qc["voices_used"] = len(by_voice)

    # The dub is uploaded and the job is already deliverable; muxing is a bonus
    # pass, so a failure here downgrades the result rather than losing it.
    if job.get("video_path") and has_column(sb, "jobs", "mp4_path"):
        try:
            muxed = mux_video(sb, job, work, out_wav)
            if muxed:
                fields["mp4_path"], mux_info = muxed
                qc["video"] = mux_info
        except Exception as e:
            traceback.print_exc()
            log(sb, job, "muxing",
                f"Video mux failed, audio is still complete: {str(e)[:200]}")

    fields["qc_summary"] = qc
    try:
        set_job(sb, job_id, **fields)
    except Exception:
        # Everything is rendered and uploaded; failing to say so would strand
        # a finished episode at 'rendering' until the stall sweep re-rendered
        # it. One fresh-token retry before treating that as a real outage.
        refresh_auth(sb, force=True)
        set_job(sb, job_id, **fields)
    _stats["jobs"] += 1
    log(sb, job, "done",
        f"{end:.1f}s rendered, coverage {fields['qc_summary']['word_coverage']}%",
        fields["qc_summary"])


# -------------------------------------------------------------------- loop

def claim_next_job(sb):
    """Atomically take one queued job. The status filter in the UPDATE is what
    stops two workers grabbing the same row."""
    q = sb.table("jobs").select("*").eq("status", "queued")
    if has_column(sb, "jobs", "priority"):
        q = q.order("priority").order("created_at")
    else:
        q = q.order("created_at")
    rows = q.limit(1).execute().data
    if not rows:
        return None
    job = rows[0]
    fields = {"status": "compiling"}
    if has_column(sb, "jobs", "claimed_at"):
        fields["claimed_at"] = now_iso()
        fields["heartbeat_at"] = now_iso()
    # render_workers only knows what a worker is on *now*, so the record of
    # which GPU rendered a job disappears the moment it finishes. Stamped here
    # instead, the queue board can still say "rendered on RunPod" a week later.
    if has_column(sb, "jobs", "worker_label"):
        fields["worker_label"] = worker_label()
        fields["worker_gpu"] = gpu_name()
    got = (sb.table("jobs").update(fields)
           .eq("id", job["id"]).eq("status", "queued").execute().data)
    return job if got else None


def claim_next_voice(sb):
    rows = (sb.table("voices").select("*").eq("status", "uploaded")
            .order("created_at").limit(1).execute().data)
    if not rows:
        return None
    v = rows[0]
    got = (sb.table("voices").update({"status": "encoding"})
           .eq("id", v["id"]).eq("status", "uploaded").execute().data)
    return v if got else None


def print_stats(t0):
    mins = (time.time() - t0) / 60
    print(f"\n--- {mins:.1f} min | {_stats['jobs']} jobs, {_stats['voices']} voices, "
          f"{_stats['cues']} cues, {_stats['audio_s']/60:.1f} min audio "
          f"({_stats['audio_s']/max(time.time()-t0,1):.2f}x realtime) ---", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true",
                    help="drain both queues and exit")
    ap.add_argument("--runtime-minutes", type=float,
                    default=float(os.environ.get("WORKER_RUNTIME_MINUTES", "0")),
                    help="stop claiming new work after this long (0 = unlimited)")
    ap.add_argument("--requeue-stalled", action="store_true",
                    help="return jobs abandoned by a dead worker to the queue first")
    args = ap.parse_args()

    sb = get_client()
    t0 = time.time()
    budget = args.runtime_minutes * 60 if args.runtime_minutes > 0 else None

    if args.requeue_stalled:
        try:
            n = sb.rpc("requeue_stalled_jobs", {"stale_minutes": 15}).execute().data
            # The sweep also returns voices rescued from a dead 'encoding'
            # claim, so the count is items, not just jobs.
            print(f"requeued {n} stalled item(s)", flush=True)
        except Exception as e:
            print(f"(requeue_stalled_jobs unavailable: {e})", flush=True)

    try:
        sb.rpc("prune_render_workers", {"older_than_hours": 72}).execute()
    except Exception:
        pass                      # optional housekeeping, never worth a failure

    beat(sb, "starting", detail="waking up", force=True)
    print(f"worker up — batch {BATCH}/{ASR_BATCH}"
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
                beat(sb, "busy", detail=f"building voice {voice['name']}", force=True)
                try:
                    process_voice(sb, voice)
                except Exception as e:
                    traceback.print_exc()
                    sb.table("voices").update({"status": "failed", "error": str(e)[:500]}) \
                        .eq("id", voice["id"]).execute()
                continue

            job = claim_next_job(sb)
            if job is not None:
                idle = 0
                print(f"\n=== job {job['id']} — {job['title']} ===", flush=True)
                beat(sb, "busy", job_id=job["id"], detail=job["title"], force=True)
                jt = time.time()
                try:
                    render_job(sb, job)
                    print(f"=== done in {time.time()-jt:.0f}s ===", flush=True)
                except Exception as e:
                    traceback.print_exc()
                    set_job(sb, job["id"], status="failed", error=str(e)[:500])
                    log(sb, job, "failed", str(e)[:300])
                print_stats(t0)
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
        # Say goodbye rather than letting the row go stale: the indicator in the
        # web app then flips to "off" the moment the Colab cell ends, instead of
        # a minute later.
        beat(sb, "stopped", detail=None, force=True)

    print_stats(t0)


if __name__ == "__main__":
    main()
