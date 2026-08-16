# GPU worker

Renders queued dub jobs and builds voice-library entries. Runs anywhere with an
NVIDIA GPU — Colab, RunPod, or a local machine.

The worker **pulls** from Supabase rather than being called by the browser. The
GPU is ephemeral (a Colab session, a spot pod) while the database is not, so a
pull-based worker survives restarts, needs no public URL or CORS, and lets
several GPUs share one queue.

```
browser ──▶ Supabase ◀── worker (here)
            queue         claims → renders → writes back
```

## Install

```bash
pip install "omnivoice[tn]" supabase
```

Optional extras, each enabling one feature and each degrading politely when
absent — a job that asks for one still renders, and says in its log why the
step was skipped:

```bash
pip install google-genai      # or `anthropic` — rewrites over-long lines
pip install demucs            # removes the original voices from a video
apt-get install -y ffmpeg     # required for video jobs at all
```

`[tn]` is text normalisation. It needs `pynini`, which has no Windows wheels but
installs fine on Linux — without it, numbers are read as digits rather than
words ("thứ 8" comes out as "thứ tư").

## Credentials

One of two sets, both read from the environment. Nothing is ever committed.

**Lovable Cloud** — no `service_role` key is issued, so the worker signs in as a
staff account and works under row-level security:

```bash
export SUPABASE_URL="https://xxxx.supabase.co"
export SUPABASE_PUBLISHABLE_KEY="sb_publishable_..."
export WORKER_EMAIL="you@example.com"
export WORKER_PASSWORD="..."
```

**Your own Supabase project** — full access, sees every user's jobs:

```bash
export SUPABASE_URL="https://xxxx.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="..."
```

Under the staff-login path the worker only sees jobs owned by that account,
unless the account is registered in `worker_accounts` (see
`migrations/20260815_shared_voice_library.sql`).

## Run

Adaptation needs an API key on the worker; without one the job renders the
original text rather than failing:

```bash
export GEMINI_API_KEY="AIza..."         # or ANTHROPIC_API_KEY
```

```bash
python worker.py                        # loop until stopped
python worker.py --once                 # drain both queues and exit
python worker.py --runtime-minutes 240  # stop claiming work after N minutes
python worker.py --requeue-stalled      # recover jobs abandoned by a dead worker
```

`--once --requeue-stalled` is the shape for a scheduled batch: boot, drain,
exit.

## Tuning

| Variable | Meaning |
|---|---|
| `WORKER_BATCH` | cues generated at once (default 4) |
| `WORKER_ASR_BATCH` | clips transcribed at once (default = `WORKER_BATCH`) |
| `OMNIVOICE_DEVICE` | default `cuda:0` |
| `WORKER_POLL_SECONDS` | idle poll interval (default 5) |
| `WORKER_LABEL` | name shown in the app (default: `Colab`, `RunPod …`, or hostname) |
| `WORKER_ID` | presence row id (default: hostname-pid) |

Measured on an 8 GB card: ASR batch 4 gives 1.52x over sequential, batch 8 only
1.58x while pushing reserved VRAM from 4.8 to 6.2 GB alongside the TTS model. On
small cards the extra 0.06x is not worth the OOM risk.

## What it writes

| Column | Meaning |
|---|---|
| `jobs.status` | `compiling → rendering → qc → assembling → done` |
| `jobs.qc_summary` | coverage %, cue counts, timing mode, overrun seconds |
| `render_workers` | presence: is this worker alive, and what is it doing |
| `jobs.srt_out_path` | corrected `.srt` matching the generated audio |
| `cues.cer` | fraction of script words **not** spoken — `0.0` is perfect |
| `cues.status` | `qc_pass`, `review`, `qc_fail`, `condensed` |
| `voices.status` | `uploaded → encoding → ready` |
| `voices.prompt_path` | cached clone prompt, so a voice is encoded once ever |

`cer` comes from real verification: every cue is synthesised, transcribed back
with Whisper, and compared against the script. Cues below 100% are retried up to
`max_attempts`, keeping the best take. Forcing a clip shorter than its words
need makes the model drop words rather than speak faster — measured 65% word
coverage when durations were hard-forced, versus 99%+ when left free.

## Presence — how the app knows the GPU is on

The worker upserts a `render_workers` row every 20 seconds, **including while
idle**, and writes `stopped` on the way out. That is what drives the engine
indicator in the app's header: a Colab session that was closed cleanly flips it
immediately, and one that was killed goes stale instead (75 s idle, 3 minutes
busy — a long voice encode legitimately blocks the beat).

`jobs.heartbeat_at` cannot substitute for this. It only ticks mid-render, so an
idle worker and a dead one look identical through it.

Presence is optional. Until the app's `20260815_engine_status.sql` migration has
been run, the worker prints `engine status off` once and renders exactly as
before.

## Job kinds

`jobs.kind` says which flow created a job, and the worker branches on it.

| kind | in | out |
|---|---|---|
| `subtitles` | an `.srt` | `dub.wav` + a corrected `.srt` |
| `video` | a video and its `.srt` | `dubbed.mp4` |
| `tts` | typed or uploaded text | audio |

A `tts` job is prose, not subtitles, so two subtitle conventions are switched
off for it — one of which loses text. Caption skipping would silently drop a
paragraph that happens to be wholly parenthesised, and `(See appendix A.)` is
content in a document. Fit-to-timecode is refused too: a reading has no edit to
honour, only the estimates this pipeline generated, so its clips simply follow
one another and the rhythm comes from the paragraph breaks in the writing.

## The tools it renders with

Each runs standalone as well as inside the worker, which is how they are tested.

| | |
|---|---|
| `speakers.py` | Finds the cast in a subtitle file — `MINH:`, `[MINH]`, italics for narration. Detection never edits the script: only the label name is recorded, and it is stripped at render time, so a wrong guess is undone by clearing it. |
| `adapt_srt.py` | Rewrites lines with more syllables than their slot can hold — the root cause of both dropped words and rushed audio. `--dry-run` reports the fit with no API key at all. |
| `video_dub.py` | Lays the dub over the source video, keeping the music and effects and ducking them under the new voice. |
| `srt_dub.py` | The original one-shot CLI: subtitles plus a reference clip in, dub out. |

```bash
python speakers.py episode.srt --show-lines
python adapt_srt.py --srt episode.srt --dry-run
python video_dub.py --video ep.mp4 --dub dub.wav --output ep_vn.mp4
```
