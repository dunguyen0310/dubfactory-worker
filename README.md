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
pip install google-genai      # or `anthropic` — rewrites over-long lines,
                              #   and translates transcribe jobs
pip install demucs            # removes the original voices from a video
apt-get install -y ffmpeg     # required for video jobs at all
```

`transcribe` jobs are the one kind with a hard dependency — without WhisperX
they cannot start at all, so install it alongside everything else rather than
after:

```bash
pip install "omnivoice[tn]" supabase whisperx
```

One `pip install` on purpose. WhisperX pins `torch~=2.8.0` and omnivoice has its
own constraints; resolving them together is what stops a second install swapping
out the torch build the first one needed. Its VAD weights ship inside the wheel,
so no Hugging Face token is involved.

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
| `WORKER_WHISPER_BATCH` | audio chunks per WhisperX batch on transcribe jobs (default 8) |
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
| `jobs.status` | `compiling → rendering → qc → assembling → done`, or `compiling → transcribing → translating → done` for a transcribe job |
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
| `transcribe` | a video or audio file | a translated `.srt` + the source transcript |

A `transcribe` job runs the pipeline backwards — it *produces* the subtitle file
the other kinds consume — and walks its own two stages,
`transcribing → translating`, rather than any of the rendering ones. The
translation is written to `cues.source_text`, the column every later stage
already reads as "the line to speak", which is what lets a finished transcript be
adapted, cast and rendered by the existing path with no special cases. What was
heard stays beside it in `cues.transcript_text`, so a questionable translation
can always be read against the original.

Its two halves resume independently, because they fail for different reasons and
cost different amounts: the transcript is committed to the database and uploaded
**before** translation is attempted, so a worker with no API key still delivers
the source transcript, and requeueing it after setting one re-runs only the
translation. `cues.translated_at` is what makes that exact — a line that is only
a name or a number legitimately translates to itself, so "has it been
translated?" cannot be inferred by comparing the two columns.

This kind does **not** degrade politely when its migration is missing. A job with
nowhere to put a transcript is refused by name up front, rather than discovered
one dropped column at a time after an hour of GPU time.

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
| `transcribe_video.py` | The reverse direction: WhisperX transcribes a video, forced alignment puts a timestamp on every *word*, those words are cut into subtitle-shaped cues, and the same provider layer `adapt_srt.py` uses translates them. `--dry-run` transcribes with no API key. |

```bash
python speakers.py episode.srt --show-lines
python adapt_srt.py --srt episode.srt --dry-run
python video_dub.py --video ep.mp4 --dub dub.wav --output ep_vn.mp4
python transcribe_video.py --video ep.mp4 --output ep_vi.srt
```

## Tests

No GPU, no model download, no API key — the model and the translation provider
are both stubbed:

```bash
python test_transcribe.py          # cue shaping: 14 checks
python test_worker_transcribe.py   # transcribe_job orchestration: 7 checks
```

Cue shaping is where the edge cases actually live, and all three of these came
out of writing those tests rather than out of review:

- **Padding may only consume silence.** Consecutive words inside a sentence are
  contiguous — one's end is the next one's start — so nudging every cue outwards
  by a fixed amount makes every cue overlap its neighbour.
- **A short line after a pause is a separate utterance.** "Yes." and "No."
  spoken a second apart are two answers; merging the second into the first
  because it is short undoes the pause break shaping just made deliberately.
- **Words the aligner could not place arrive with the keys missing**, not null —
  numerals and foreign names do it constantly. They still have to be spoken, so
  they take the timing of whatever surrounds them rather than being dropped.

The orchestration tests run `transcribe_job` against a fake Supabase, and the
property they exist to hold is that a requeued job never pays for ASR twice.
