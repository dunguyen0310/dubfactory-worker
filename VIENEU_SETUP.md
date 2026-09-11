# VieNeu Studio — the second engine

`vieneu_worker.py` drains a **separate queue** from `worker.py`, using
[VieNeu-TTS v3 Turbo](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo)
rather than OmniVoice. It powers the **Studio** tab in the web app.

The two pipelines share a Supabase project and nothing else — different tables,
different buckets, different worker process, different model. A broken
OmniVoice install cannot stop the studio and a broken VieNeu install cannot
stop an episode.

|  | Dub pipeline (`worker.py`) | Studio (`vieneu_worker.py`) |
|---|---|---|
| Model | OmniVoice, 600+ languages | VieNeu-TTS v3 Turbo, Vietnamese + English |
| Sample rate | 24 kHz | **48 kHz** |
| Hardware | **needs an NVIDIA GPU** | **CPU is fine** — ONNX, torch-free |
| Voices | cloned or designed, all built by a worker | **23 built-in presets** (20 before vieneu 3.6.4) + instant cloning |
| Input | subtitles, video, documents | typed text |
| Timecode | cues must fit their slots | none — nothing to fit |
| Tables | `jobs`, `cues`, `voices`, `render_workers` | `vieneu_jobs`, `vieneu_lines`, `vieneu_voices`, `vieneu_workers` |
| Buckets | `srt`, `videos`, `voices`, `outputs` | `vieneu`, `vieneu-outputs` |

**The CPU line is the important one.** Nothing in the studio needs Colab. It
runs on the laptop that is already open, faster than realtime.

## Setup

### 1. Run the migration

`migrations/20260821_vieneu_studio.sql` (in the app repo), in the Lovable SQL
editor. It creates the four tables, both buckets, the RLS policies and two
housekeeping functions. Safe to re-run.

Unlike most of the dub pipeline's features, this one does **not** degrade
politely without it — a studio with nowhere to put a job is not a studio. The
app says so by name in the engine popover rather than failing at the first
click.

### 2. Install

```bash
pip install vieneu supabase soundfile numpy
```

That is the CPU install: ONNX Runtime, no torch, ~1 GB of weights downloaded on
first run. For the GPU path (batched throughput on long jobs — not needed
otherwise):

```bash
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==4.57.6" vieneu supabase
```

> Install VieNeu in **its own environment**, not the one holding OmniVoice.
>
> The CPU build is torch-free and pulls no `transformers` at all — verified on
> 3.3.0: `onnxruntime`, and neither package anywhere in site-packages. What it
> *does* bring is its own `tokenizers`, `huggingface_hub`, `numba`, `librosa`
> and `gradio`, which is very nearly the exact set the dub stack's install is
> most delicate about (see the comments in the Colab notebook's install cell for
> what that delicacy cost to find). Dropping five more pinned packages into that
> resolution and hoping pip reconciles them is a poor trade against 750 MB of
> disk and two minutes.
>
> The **GPU** build is a harder no: it pins `transformers==4.57.6` and omnivoice
> needs `>=5.3`, so those two genuinely cannot share an environment at all.
>
> Two virtualenvs, two processes, no shared dependency — which is the same
> reason the tables are separate.

### 3. Run

```bash
export SUPABASE_URL="https://<project>.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="<key>"        # or the Lovable Cloud pair below
python vieneu_worker.py
```

On Lovable Cloud there is no service-role key, so the worker signs in as a
staff account and works under RLS — the same arrangement `WORKER_SETUP.md`
describes for the dub queue:

```bash
export SUPABASE_URL="https://<project>.supabase.co"
export SUPABASE_PUBLISHABLE_KEY="sb_publishable_..."
export WORKER_EMAIL="worker@example.com"
export WORKER_PASSWORD="..."
```

It only sees that account's jobs, so queue from the same login.

```bash
python vieneu_worker.py                       # loop until stopped
python vieneu_worker.py --once                # drain both queues and exit
python vieneu_worker.py --list-voices         # what the model actually knows
python vieneu_worker.py --requeue-stalled     # rescue jobs a dead worker held
```

Environment:

| Variable | Default | Meaning |
|---|---|---|
| `VIENEU_BACKEND` | `onnx` | `onnx` (CPU, streaming path) or `pytorch` (CUDA, batching) |
| `VIENEU_PRECISION` | `int8` | `int8` is ~1.6× faster and ~4× smaller; `fp32` for a final master |
| `VIENEU_DEVICE` | `cuda:0` | only consulted on the pytorch backend |
| `VIENEU_POLL_SECONDS` | `5` | idle poll interval |
| `VIENEU_WORKER_ID` | hostname-pid | presence row id |
| `VIENEU_WORKER_LABEL` | hostname / Colab / RunPod | what the team sees in the app |

Asking for `pytorch` on a machine with no CUDA **falls back to ONNX** rather
than failing every job, and the presence row records what it actually ended up
running — so the app shows the truth rather than the request.

### Verified against vieneu 3.6.4 (September 2026)

The signatures the worker calls — `Vieneu()`, `infer`, `infer_batch`,
`add_voice(denoise=…)`, `denoise`, `list_preset_voices`, `save` — are unchanged
from 3.3.0. The only additions are a `babble_retries=2` constructor knob and a
`mode="v3nano"` variant: a 48M-parameter model at 24 kHz, roughly 3× faster and
noticeably worse on English and code-switched text. The worker does not use it;
the Studio is sold on 48 kHz.

Measured on an i9-10900K (Comet Lake, no VNNI), so this is the pessimistic
case for int8:

| precision | model load | speed | note |
|---|---|---|---|
| `int8` (default) | 8 s | **3.2× realtime** | still faster without VNNI |
| `fp32` | 22 s | 2.0× realtime | for a final master, if ever |

### Running it on a Windows machine

```powershell
py -3.13 -m venv .venv-vieneu          # 3.10–3.13; the 3.14 launcher default is too new
.\.venv-vieneu\Scripts\pip.exe install vieneu supabase soundfile numpy
.\.venv-vieneu\Scripts\python.exe vieneu_worker.py --list-voices   # downloads ~1 GB once
```

Two Windows-only notes. Hugging Face warns that the cache "uses symlinks by
default" and your machine "does not support them"; the cache still works, it
just duplicates a file or two. Enabling Developer Mode in Settings silences it.
And the worker widens stdout to UTF-8 on start-up because a Windows console is
cp1252 by default and every preset is named in Vietnamese — the first version
died printing a voice name, after the render had already finished.

## The three job kinds

| `kind` | In | Out |
|---|---|---|
| `speak` | one script | one continuous file (`audio_path`) |
| `batch` | many lines | one file per line, a `.zip`, and a joined audition |
| `denoise` | a recording | the same recording, cleaned. No synthesis, no voice |

`speak` is chunked by the app before it is queued, one `vieneu_lines` row per
chunk. That is not an optimisation — the model chunks internally regardless
(`max_chars=256`) — it is what makes the chunk count honest on screen, gives
every chunk its own audition, and lets one bad sentence be re-rolled without
re-rendering the other ninety.

## Voices

The presets ship **inside the model**, addressed by name (`Adam`, `Xuân
Vĩnh`). They are not rows in any table: copying them into one would create a
second source of truth that goes stale the moment the model ships a new voice —
which it did: vieneu 3.6.4 went from 20 to 23 (`Mạnh Dũng`, `Minh Quân`, `Anh
Khôi`, all northern and male) and moved its default from `Adam` to `Minh Quân`.
Nothing here needed changing for that; the worker asks the model. `vieneu_jobs.voice_preset` holds the name, and the worker checks it
against `Vieneu.list_preset_voices()` — an unknown name fails the job **saying
what the model does know**, rather than quietly falling back to a default.

Cloned voices are rows (`vieneu_voices`), and building one is instant: no
training step, no GPU. What the worker actually spends time on is the
**audition** it renders afterwards, and that is the deliverable — cloning never
reports a quality score, so hearing the result is the only way to know whether
a reference clip worked.

`add_voice` is per-process state, so a restart re-enrolls. The worker caches
enrolments for the session, which is why a hundred lines in one cloned voice
encode the reference once rather than a hundred times.

Consent is enforced **on the worker as well as in the form**: a voice can be
added by one person and used by another, and the second person is the one worth
stopping.

## Settings

`vieneu_jobs.settings` is jsonb and passes through. The studio's own controls
write these:

```jsonc
{
  "backend": "onnx",       // onnx | pytorch — a worker can only run what it has
  "precision": "int8",     // int8 | fp32
  "temperature": 0.8,      // the model's stability sweet spot
  "batch_size": 32,        // pytorch only; 1 disables batching
  "join_gap_ms": 220,      // silence between chunks when a speak job is joined
  "format": "wav"          // wav (48 kHz lossless) | mp3
}
```

Anything else in `MODEL_KWARGS` also reaches the model with no worker change:
`top_k`, `top_p`, `max_new_frames`, `repetition_penalty`, `repetition_window`,
`max_chars`, `silence_p`, `crossfade_p`, `use_ref_codes`, `apply_watermark`.

> **v3 Turbo watermarks its output by default** (`apply_watermark=True` inside
> the model). This worker leaves that alone. A job that sets it false has made
> that choice deliberately.

Two of the settings above are requests rather than instructions. `backend` and
`precision` describe what the *worker* was started with — a CPU-only worker
cannot honour `"pytorch"` — so whichever was really used is written to
`qc_summary` on the finished job.

## Resuming, re-rolling and failure

- **Every line's clip is kept.** A requeued job re-renders only lines marked
  `rerolled`, never rendered, or whose file is missing from storage. Re-rolling
  one sentence of a 200-line batch costs one sentence.
- **A line marked done but missing its audio is re-rendered.** Storage and
  Postgres can disagree; trusting the row alone would assemble a file with a
  hole in it.
- **One bad line does not fail the job.** It is recorded on the line, the good
  lines are still assembled, and `qc_summary.failed` says how many. A job where
  *every* line failed does fail — there is nothing to hand over.
- **A failed batch retries line by line.** A GPU batch is one forward pass and
  fails as a unit, so one unspeakable line would otherwise take its whole group.
- **A worker killed mid-job** leaves the job at `running`; `requeue_stalled_vieneu()`
  returns it after 10 minutes of silence. Much shorter than the dub queue's 15
  because work here is seconds-to-minutes, so silence means death rather than
  a long render.

## Testing it without the app

```bash
python test_vieneu_worker.py     # 38 checks, no GPU, no network, no model
python vieneu_worker.py --list-voices
```

The test suite stubs the model with something that emits a recognisable tone
per line, so the assembled file is checked for real rather than asserted about.
It covers the parts that go wrong quietly: which lines a requeued job
re-renders, whether one bad line takes the others with it, whether a batch ever
straddles two voices, and whether consent is enforced.

## Known sharp edges

- **`denoise()` returns 44.1 kHz**, not the 48 the synthesiser emits. The
  worker reads the rate back from the file rather than assuming it, and records
  it in `qc_summary.sample_rate`.
- **MP3 needs libsndfile ≥ 1.1.** Where it is older the worker writes a WAV,
  says so in the log, and labels it honestly — the audio is already rendered by
  then, and a WAV nobody asked for still plays.
- **Emotion cues are experimental** per the model card: `[cười]`, `[thở dài]`,
  `[hắng giọng]`. They work, but not every time.
- **`style` is deprecated** on v3 Turbo and ignored. The reading style is baked
  into the voice, which is why the picker has a Character column and no style
  control.
