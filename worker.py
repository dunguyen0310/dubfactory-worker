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

Run:
    python worker.py                       # loop until stopped
    python worker.py --once                # drain both queues then exit
    python worker.py --runtime-minutes 200 # stop cleanly before a Colab session ends
"""

import argparse
import os
import re
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

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


def refresh_auth(sb):
    """Re-sign-in before the JWT expires. A dub run lasts hours; the default
    access token does not."""
    if not _auth["email"]:
        return
    if time.time() - _auth["at"] < 40 * 60:
        return
    try:
        sb.auth.sign_in_with_password({"email": _auth["email"],
                                       "password": _auth["password"]})
        _auth["at"] = time.time()
    except Exception as e:
        print(f"  (token refresh failed: {e})", flush=True)


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


def transcribe_many(model, clips: list[np.ndarray]) -> list[str]:
    """Transcribe a list of clips, batched through the underlying HF pipeline.

    model.transcribe() handles one clip per call; the pipeline it wraps accepts
    a list, which keeps the GPU busy instead of paying per-call overhead on
    every short cue. Falls back to one-at-a-time if anything about the batched
    path fails, since verification must never be the thing that breaks a job.
    """
    if not clips:
        return []
    try:
        inputs = [{"array": c.astype(np.float32), "sampling_rate": SR} for c in clips]
        out = model._asr_pipe(inputs, batch_size=max(1, ASR_BATCH))
        if isinstance(out, dict):
            out = [out]
        return [str(o["text"]).strip() for o in out]
    except Exception as e:
        print(f"  (batched ASR unavailable, falling back: {e})", flush=True)
        import torch
        texts = []
        for c in clips:
            t = model.transcribe((torch.from_numpy(c), SR))
            texts.append(str(t[0] if isinstance(t, (list, tuple)) else t).strip())
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

def render_job(sb, job):
    job_id, owner = job["id"], job["owner_id"]
    settings = job.get("settings") or {}
    read_captions = bool(settings.get("read_caption_cues", False))
    timing_mode = settings.get("timing_mode", "natural")      # natural | fit
    language = settings.get("language", "Vietnamese")
    max_attempts = int(settings.get("max_attempts", 4))

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

    prompt, ref_timbre = get_voice_prompt(sb, vrow, work)
    model = get_model()

    set_job(sb, job_id, status="rendering", total_cues=len(cues), done_cues=len(skipped))
    log(sb, job, "rendering", f"Rendering {len(speak)} cue(s) as {vrow['name']}")

    clips: dict[int, np.ndarray] = {}
    done, review = len(skipped), 0
    beat = time.time()

    for start in range(0, len(speak), BATCH):
        batch = speak[start:start + BATCH]
        best = {c["id"]: None for c in batch}
        pending = list(batch)

        for attempt in range(1, max_attempts + 1):
            kw = {"language": language}
            if attempt > 1:
                kw["duration"] = [
                    S.natural_estimate(c["source_text"]) * (1 + 0.1 * (attempt - 1)) / S.OVERSHOOT
                    for c in pending
                ]
            outs = model.generate(
                text=[c["source_text"] for c in pending],
                voice_clone_prompt=[prompt] * len(pending), **kw,
            )
            wavs = [np.asarray(a, dtype=np.float32).squeeze() for a in outs]
            heard = transcribe_many(model, wavs)

            still = []
            for c, wav, txt in zip(pending, wavs, heard):
                cov, missing, tonal = S.coverage(c["source_text"], txt)
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
            wav, cov, sim, missing, tonal = best[c["id"]]
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
                sb.storage.from_("outputs").upload(
                    path, local.read_bytes(),
                    {"content-type": "audio/wav", "upsert": "true"})
            except Exception as e:
                print(f"  cue {c['idx']} upload failed: {e}", flush=True)
                path = None
            done += 1
            sb.table("cues").update({
                "status": status, "cer": cer, "audio_path": path,
                "rendered_ms": int(wav.size / SR * 1000),
                "final_text": c["source_text"], "note": note,
            }).eq("id", c["id"]).execute()

        fields = {"done_cues": done, "review_cues": review}
        if time.time() - beat > 30 and has_column(sb, "jobs", "heartbeat_at"):
            fields["heartbeat_at"] = now_iso()
            beat = time.time()
        set_job(sb, job_id, **fields)

    # ------------------------------------------------------------ assemble
    set_job(sb, job_id, status="qc")
    log(sb, job, "qc", f"{done - review}/{done} cues passed verification")
    set_job(sb, job_id, status="assembling")

    placed, cursor = [], 0.0
    for c in sorted(speak, key=lambda x: x["idx"]):
        clip = clips.get(c["idx"])
        if clip is None:
            continue
        start_s = max(c["start_ms"] / 1000.0, cursor)
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
    sb.storage.from_("outputs").upload(
        wav_path, out_wav.read_bytes(),
        {"content-type": "audio/wav", "upsert": "true"})

    fields = {"status": "done", "wav_path": wav_path, "error": None,
              "done_cues": done, "review_cues": review}

    # Natural-mode audio runs longer than the source subtitles, so a corrected
    # .srt is what makes the result usable in an editor.
    if has_column(sb, "jobs", "srt_out_path"):
        lines = []
        for n, (c, a, b) in enumerate(final_times, 1):
            lines.append(f"{n}\n{S.fmt_ts(a)} --> {S.fmt_ts(b)}\n{c['source_text']}\n")
        srt_path = f"{owner}/{job_id}/dub.srt"
        sb.storage.from_("outputs").upload(
            srt_path, ("\n".join(lines)).encode("utf-8"),
            {"content-type": "application/x-subrip", "upsert": "true"})
        fields["srt_out_path"] = srt_path

    cer_rows = sb.table("cues").select("cer").eq("job_id", job_id).execute().data
    cers = [r["cer"] for r in cer_rows if r.get("cer") is not None]
    fields["qc_summary"] = {
        "word_coverage": round((1 - float(np.mean(cers))) * 100, 2) if cers else 0.0,
        "cues_total": len(cues), "cues_spoken": len(placed),
        "cues_skipped": len(skipped), "cues_needing_review": review,
        "timing_mode": timing_mode, "speed_applied": round(speed, 3),
        "audio_seconds": round(end, 2), "srt_seconds": round(srt_end, 2),
        "overrun_seconds": round(max(0.0, end - srt_end), 2),
        "voice": vrow["name"],
    }
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
            print(f"requeued {n} stalled job(s)", flush=True)
        except Exception as e:
            print(f"(requeue_stalled_jobs unavailable: {e})", flush=True)

    print(f"worker up — batch {BATCH}/{ASR_BATCH}"
          + (f", budget {args.runtime_minutes:.0f} min" if budget else "")
          + " (Ctrl+C to stop)", flush=True)

    idle = 0
    while True:
        if budget and time.time() - t0 > budget:
            print("\nruntime budget reached — stopping cleanly", flush=True)
            break
        refresh_auth(sb)

        voice = claim_next_voice(sb)
        if voice is not None:
            idle = 0
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
        time.sleep(POLL)

    print_stats(t0)


if __name__ == "__main__":
    main()
