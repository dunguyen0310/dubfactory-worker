"""Orchestration tests for vieneu_worker, against a fake Supabase and a fake model.

The model itself cannot run here — no `vieneu` install and no 1 GB of weights —
but the parts that go wrong quietly can: which lines a requeued job re-renders,
whether one bad line takes the other ninety-nine with it, whether a cloned voice
with no consent is refused, and whether the assembled file is the lines in the
right order.

`get_model` is stubbed with something that emits a recognisable tone per line,
so the joined result can be checked for real rather than asserted about:

    python test_vieneu_worker.py

Needs numpy and soundfile, which the worker needs anyway. No GPU, no network,
no API key.
"""

import re
import sys


import numpy as np
import soundfile as sf

import vieneu_worker as V


SR = 48_000


# ------------------------------------------------------------- fake supabase

# Nullable columns the real tables have, so an inserted row looks like one that
# came back from Postgres rather than only the keys the test happened to set.
COLUMN_DEFAULTS = {
    "vieneu_jobs": ("body", "voice_preset", "voice_id", "audio_path", "zip_path",
                    "source_path", "duration_ms", "error", "qc_summary",
                    "claimed_at", "heartbeat_at", "finished_at"),
    "vieneu_lines": ("voice_preset", "voice_id", "audio_path", "duration_ms",
                     "error"),
    "vieneu_voices": ("sample_path", "preview_path", "error", "qc"),
}


class Result:
    def __init__(self, data):
        self.data = data


class Query:
    def __init__(self, db, table, op, payload=None):
        self.db, self.table, self.op, self.payload = db, table, op, payload
        self.filters = {}

    def select(self, *a, **k):
        return self

    def eq(self, col, val):
        self.filters[col] = val
        return self

    def order(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def _match(self, row):
        return all(row.get(k) == v for k, v in self.filters.items())

    def execute(self):
        rows = self.db.tables.setdefault(self.table, [])
        if self.op == "select":
            hit = [r for r in rows if self._match(r)]
            hit.sort(key=lambda r: r.get("idx", 0))
            return Result(hit)
        if self.op in ("insert", "upsert"):
            new = self.payload if isinstance(self.payload, list) else [self.payload]
            for r in new:
                r = dict(r)
                cur = (next((x for x in rows if x.get("id") == r.get("id")), None)
                       if self.op == "upsert" and r.get("id") is not None else None)
                if cur is not None:
                    cur.update(r)
                    continue
                r.setdefault("id", f"{self.table}-{len(rows) + 1}")
                for col in COLUMN_DEFAULTS.get(self.table, ()):
                    r.setdefault(col, None)
                rows.append(r)
            self.db.log.append((self.table, self.op, len(new)))
            return Result(new)
        if self.op == "update":
            hit = [r for r in rows if self._match(r)]
            for r in hit:
                r.update(self.payload)
            if self.table == "vieneu_jobs" and "status" in self.payload:
                self.db.statuses.append(self.payload["status"])
            self.db.log.append((self.table, "update", tuple(self.payload)))
            return Result(hit)
        if self.op == "delete":
            hit = [r for r in rows if self._match(r)]
            for r in hit:
                rows.remove(r)
            return Result(hit)
        raise AssertionError(f"unhandled op {self.op}")


class Bucket:
    def __init__(self, db, name):
        self.db, self.name = db, name

    def download(self, path):
        store = self.db.files.get(self.name, {})
        if path not in store:
            raise RuntimeError(f"{self.name}/{path} not found")
        return store[path]

    def upload(self, path, data, opts=None):
        self.db.files.setdefault(self.name, {})[path] = data
        self.db.uploads.append(path)
        return Result([{"path": path}])


class Storage:
    def __init__(self, db):
        self.db = db

    def from_(self, name):
        return Bucket(self.db, name)


class FakeDB:
    def __init__(self):
        self.tables = {"vieneu_jobs": [], "vieneu_lines": [],
                       "vieneu_voices": [], "vieneu_workers": []}
        self.files = {V.BUCKET_IN: {}, V.BUCKET_OUT: {}}
        self.storage = Storage(self)
        self.uploads = []
        self.statuses = []
        self.log = []

    def table(self, name):
        return _TableHandle(self, name)

    def rpc(self, *a, **k):
        return Query(self, "rpc", "select")


class _TableHandle:
    def __init__(self, db, name):
        self.db, self.name = db, name

    def select(self, *a, **k):
        return Query(self.db, self.name, "select")

    def insert(self, payload):
        return Query(self.db, self.name, "insert", payload)

    def upsert(self, payload):
        return Query(self.db, self.name, "upsert", payload)

    def update(self, payload):
        return Query(self.db, self.name, "update", payload)

    def delete(self):
        return Query(self.db, self.name, "delete")


# ---------------------------------------------------------------- fake model

class FakeModel:
    """Emits a per-voice tone so a joined file can be checked, not just counted.

    `fail_on` makes one line raise, which is the case the real model hits on a
    text it cannot phonemise and the one that must not take the job with it.
    """

    PRESETS = [("Adam", "Adam"), ("Phạm Tuyên", "Phạm Tuyên"),
               ("Xuân Vĩnh", "Xuân Vĩnh")]

    def __init__(self, fail_on=(), seconds=0.25, save_fails_after=0):
        self.fail_on = set(fail_on)
        self.seconds = seconds
        self.save_fails_after = save_fails_after
        self.saves = 0
        self.spoken = []          # (text, voice), in call order
        self.enrolled = []        # (key, path, kwargs)
        self.batches = []         # (texts, voice, batch_size, kwargs) per call
        self.gen_kwargs = []      # generation knobs each infer() actually got

    def list_preset_voices(self):
        return list(self.PRESETS)

    def infer(self, text, voice=None, **kwargs):
        if text in self.fail_on:
            raise RuntimeError("cannot phonemise that")
        self.spoken.append((text, voice))
        self.gen_kwargs.append(kwargs)
        n = int(SR * self.seconds)
        t = np.arange(n, dtype=np.float32) / SR
        # One distinguishable frequency per voice, so ordering is verifiable.
        freq = 200 + 100 * (abs(hash(voice)) % 5)
        return 0.2 * np.sin(2 * np.pi * freq * t).astype(np.float32)

    def infer_batch(self, texts, voice=None, batch_size=None, **kwargs):
        # A real batch is one forward pass and fails as a unit, which is the
        # behaviour that matters here.
        self.batches.append((list(texts), voice, batch_size, kwargs))
        if any(t in self.fail_on for t in texts):
            raise RuntimeError("batch contained an unspeakable line")
        return [self.infer(t, voice=voice, **kwargs) for t in texts]

    def save(self, audio, path):
        # `save_fails_after` reproduces a batch that dies part way through
        # persisting: the earlier lines are already banked.
        self.saves += 1
        if self.save_fails_after and self.saves > self.save_fails_after:
            self.save_fails_after = 0        # only once, so the retry can pass
            raise RuntimeError("disk went away mid-batch")
        sf.write(path, np.asarray(audio, dtype=np.float32), SR,
                 format="WAV", subtype="PCM_16")

    def add_voice(self, key, path, denoise=True):
        self.enrolled.append((key, path, {"denoise": denoise}))

    def denoise(self, src, out_path=None):
        data, sr = sf.read(src, dtype="float32", always_2d=False)
        quiet = (data * 0.25).astype(np.float32)
        if out_path:
            sf.write(out_path, quiet, sr, format="WAV", subtype="PCM_16")
        return quiet, sr


def use_model(model, backend="onnx"):
    """Point the worker at a fake and reset the per-process caches it keeps."""
    V._model = model
    V._model_info.clear()
    V._model_info.update(backend=backend, precision="int8",
                         device="cuda:0" if backend == "pytorch" else "cpu")
    V._enrolled.clear()
    V._presets.clear()
    V._presets.update(vid for _, vid in model.list_preset_voices())
    V._beat["on"] = False        # presence is not what these tests are about
    V._touch["at"] = 9e9         # and neither is the heartbeat
    for k in V._stats:
        V._stats[k] = 0 if k != "audio_s" else 0.0
    return model


# ------------------------------------------------------------------ fixtures

def seed_job(db, *, kind="speak", lines, preset="Adam", voice_id=None,
             settings=None, statuses=None, with_audio=()):
    job = {
        "id": "job-1", "owner_id": "owner-1", "kind": kind, "title": "Test",
        "body": "\n".join(lines), "voice_preset": preset, "voice_id": voice_id,
        "settings": settings or {}, "status": "running",
        "total_lines": len(lines), "done_lines": 0, "priority": 5,
    }
    db.table("vieneu_jobs").insert(job).execute()
    for i, body in enumerate(lines):
        row = {"id": f"line-{i}", "owner_id": "owner-1", "job_id": "job-1",
               "idx": i, "body": body,
               "status": (statuses or {}).get(i, "pending")}
        if i in with_audio:
            path = f"owner-1/job-1/{i:04d}.wav"
            row["audio_path"] = path
            tone = np.full(int(SR * 0.1), 0.5, dtype=np.float32)
            import io
            buf = io.BytesIO()
            sf.write(buf, tone, SR, format="WAV", subtype="PCM_16")
            db.files[V.BUCKET_OUT][path] = buf.getvalue()
        db.table("vieneu_lines").insert(row).execute()
    return db.tables["vieneu_jobs"][0]


def wav_seconds(data: bytes) -> float:
    import io
    info = sf.info(io.BytesIO(data))
    return info.frames / info.samplerate


checks = []


def check(fn):
    checks.append((fn.__name__.replace("_", " "), fn))
    return fn


# -------------------------------------------------------------------- audio

@check
def join_inserts_the_gap_and_keeps_order():
    a = np.full(SR // 10, 0.5, dtype=np.float32)
    b = np.full(SR // 10, -0.5, dtype=np.float32)
    out = V.join([a, b], SR, 200)
    assert len(out) == len(a) + len(b) + SR // 5, len(out)
    assert out[0] == 0.5 and out[-1] == -0.5
    # The gap is silence, and it is between them rather than at either end.
    assert abs(out[len(a) + 100]) < 1e-6


@check
def join_of_one_clip_adds_no_gap():
    a = np.full(1000, 0.5, dtype=np.float32)
    assert len(V.join([a], SR, 500)) == 1000


@check
def join_of_nothing_is_empty_not_a_crash():
    assert len(V.join([], SR, 200)) == 0


@check
def dbfs_handles_digital_silence():
    # A real input: an empty take, or a reference clip that is all zeros. log10
    # of zero would be -inf and poison every arithmetic comparison downstream.
    assert V.dbfs(np.zeros(1000, dtype=np.float32)) == -120.0
    assert V.dbfs(np.array([], dtype=np.float32)) == -120.0


@check
def dbfs_reads_a_known_level():
    full = np.ones(1000, dtype=np.float32)
    assert abs(V.dbfs(full)) < 0.01, V.dbfs(full)
    half = np.full(1000, 0.5, dtype=np.float32)
    assert abs(V.dbfs(half) + 6.02) < 0.05, V.dbfs(half)


@check
def encode_round_trips_a_wav():
    tone = np.sin(np.linspace(0, 100, SR)).astype(np.float32)
    data, ext, mime = V.encode(tone, SR, "wav")
    assert (ext, mime) == ("wav", "audio/wav")
    assert abs(wav_seconds(data) - 1.0) < 0.01


@check
def encode_falls_back_to_wav_when_mp3_is_unavailable():
    # libsndfile only grew MP3 writing in 1.1. Whichever this build is, the
    # call must produce a playable file and label it honestly rather than raise.
    tone = np.sin(np.linspace(0, 100, SR // 2)).astype(np.float32)
    data, ext, mime = V.encode(tone, SR, "mp3")
    assert ext in ("mp3", "wav"), ext
    assert mime == ("audio/mpeg" if ext == "mp3" else "audio/wav")
    assert len(data) > 100


# --------------------------------------------------------------- signatures

@check
def supported_reads_a_real_signature():
    def f(a, b=1, *, denoise=True):
        pass
    got = V.supported(f, "denoise", "temperature")
    assert got == {"denoise": True, "temperature": False}, got


@check
def supported_assumes_yes_for_kwargs_catchalls():
    # A wrapped or **kwargs-forwarding method has no named parameters to find,
    # and refusing to pass anything to it would silently drop every knob.
    def f(a, **kwargs):
        pass
    assert V.supported(f, "denoise") == {"denoise": True}


@check
def supported_says_no_when_it_cannot_introspect():
    assert V.supported(len, "denoise") == {"denoise": False}


@check
def enrol_key_cannot_collide_with_a_preset():
    # Two people can both name a voice "Adam". The row id cannot be either.
    key = V.enrol_key("11111111-2222-3333-4444-555555555555")
    assert key not in {vid for _, vid in FakeModel.PRESETS}
    assert key != "Adam"


# ------------------------------------------------------------- speak / batch

@check
def speak_renders_every_line_and_joins_them():
    db, model = FakeDB(), use_model(FakeModel(seconds=0.25))
    job = seed_job(db, lines=["Một.", "Hai.", "Ba."],
                   settings={"join_gap_ms": 200})
    V.speech_job(db, job)

    assert [t for t, _ in model.spoken] == ["Một.", "Hai.", "Ba."], model.spoken
    row = db.tables["vieneu_jobs"][0]
    assert row["status"] == "done", row
    assert row["done_lines"] == 3 and row["total_lines"] == 3
    # Three quarter-second clips and two 200 ms gaps.
    assert abs(wav_seconds(db.files[V.BUCKET_OUT][row["audio_path"]]) - 1.15) < 0.02
    # Per-line clips are kept, which is what makes a re-roll cheap later.
    assert sum(1 for p in db.uploads if p.endswith(".wav") and "/00" in p) == 3


@check
def speak_produces_no_zip():
    # A continuous read is one file. A zip of its chunks would be an invitation
    # to hand someone the pieces of a thing that was supposed to be whole.
    db, model = FakeDB(), use_model(FakeModel())
    job = seed_job(db, lines=["Một.", "Hai."])
    V.speech_job(db, job)
    assert db.tables["vieneu_jobs"][0]["zip_path"] is None


@check
def batch_writes_a_zip_and_a_joined_audition():
    db, model = FakeDB(), use_model(FakeModel())
    job = seed_job(db, kind="batch", lines=["Một.", "Hai.", "Ba."])
    V.speech_job(db, job)

    row = db.tables["vieneu_jobs"][0]
    assert row["zip_path"], row
    import io, zipfile
    z = zipfile.ZipFile(io.BytesIO(db.files[V.BUCKET_OUT][row["zip_path"]]))
    assert z.namelist() == ["0001.wav", "0002.wav", "0003.wav"], z.namelist()
    # And the joined file too, so twenty auditions are one press to listen to.
    assert row["audio_path"] and row["audio_path"] in db.files[V.BUCKET_OUT]


@check
def a_requeued_job_re_renders_only_the_rerolled_line():
    """The whole point of keeping per-line clips.

    Three lines are already done with audio in storage and one is marked
    `rerolled`. Only that one may reach the model — re-rolling one sentence of
    a long script has to cost one sentence.
    """
    db, model = FakeDB(), use_model(FakeModel())
    job = seed_job(
        db, lines=["Một.", "Hai.", "Ba.", "Bốn."],
        statuses={0: "done", 1: "done", 2: "rerolled", 3: "done"},
        with_audio=(0, 1, 3),
    )
    V.speech_job(db, job)

    assert [t for t, _ in model.spoken] == ["Ba."], model.spoken
    row = db.tables["vieneu_jobs"][0]
    assert row["status"] == "done"
    assert row["done_lines"] == 4, row["done_lines"]
    # All four are in the assembled file, not just the one that was re-rendered.
    assert abs(wav_seconds(db.files[V.BUCKET_OUT][row["audio_path"]])
               - (0.1 * 3 + 0.25 + 0.22 * 3)) < 0.05


@check
def a_line_marked_done_but_missing_its_audio_is_re_rendered():
    # Storage and Postgres can disagree — a sweep that deleted too much, an
    # upload that failed after the row was written. Trusting the row alone
    # would assemble a file with a hole in it.
    db, model = FakeDB(), use_model(FakeModel())
    job = seed_job(db, lines=["Một.", "Hai."], statuses={0: "done", 1: "done"},
                   with_audio=(0,))
    V.speech_job(db, job)
    assert [t for t, _ in model.spoken] == ["Hai."], model.spoken


@check
def one_failing_line_does_not_fail_the_job():
    db, model = FakeDB(), use_model(FakeModel(fail_on=["Hai."]))
    job = seed_job(db, lines=["Một.", "Hai.", "Ba."])
    V.speech_job(db, job)

    row = db.tables["vieneu_jobs"][0]
    assert row["status"] == "done", row
    assert "1 line(s) failed" in (row["error"] or ""), row["error"]
    assert row["qc_summary"]["failed"] == 1
    lines = {r["idx"]: r for r in db.tables["vieneu_lines"]}
    assert lines[1]["status"] == "failed" and lines[1]["error"]
    assert lines[0]["status"] == "done" and lines[2]["status"] == "done"
    # The good lines are still assembled rather than thrown away with the bad.
    assert row["audio_path"] in db.files[V.BUCKET_OUT]


@check
def a_job_where_every_line_fails_is_a_failure():
    db, model = FakeDB(), use_model(FakeModel(fail_on=["Một.", "Hai."]))
    job = seed_job(db, lines=["Một.", "Hai."])
    try:
        V.speech_job(db, job)
    except RuntimeError as e:
        assert "nothing to assemble" in str(e).lower(), e
    else:
        raise AssertionError("a job with no audio at all should not report done")


@check
def an_unknown_preset_is_named_rather_than_guessed():
    db, model = FakeDB(), use_model(FakeModel())
    job = seed_job(db, lines=["Một."], preset="Adem")
    try:
        V.speech_job(db, job)
    except RuntimeError as e:
        assert "Adem" in str(e), e
        assert "Adam" in str(e), "the message should say what the model does know"
    else:
        raise AssertionError("an unknown voice must not fall back to a default")
    assert model.spoken == []


@check
def a_per_line_voice_overrides_the_jobs():
    db, model = FakeDB(), use_model(FakeModel())
    job = seed_job(db, kind="batch", lines=["Một.", "Hai."], preset="Adam")
    db.tables["vieneu_lines"][1]["voice_preset"] = "Xuân Vĩnh"
    V.speech_job(db, job)
    assert model.spoken == [("Một.", "Adam"), ("Hai.", "Xuân Vĩnh")], model.spoken


@check
def the_gpu_backend_batches_lines_that_share_a_voice():
    db, model = FakeDB(), use_model(FakeModel(), backend="pytorch")
    job = seed_job(db, kind="batch", lines=["Một.", "Hai.", "Ba."],
                   settings={"batch_size": 16})
    V.speech_job(db, job)

    assert len(model.batches) == 1, model.batches
    texts, voice, size, _ = model.batches[0]
    assert texts == ["Một.", "Hai.", "Ba."] and voice == "Adam" and size == 16
    assert db.tables["vieneu_jobs"][0]["done_lines"] == 3


@check
def the_cpu_backend_does_not_batch():
    # `infer_batch` runs sequentially on ONNX anyway, so going through it there
    # would trade away per-line error isolation for nothing.
    db, model = FakeDB(), use_model(FakeModel(), backend="onnx")
    job = seed_job(db, kind="batch", lines=["Một.", "Hai.", "Ba."])
    V.speech_job(db, job)
    assert model.batches == [], model.batches
    assert len(model.spoken) == 3


@check
def a_batch_never_straddles_two_voices():
    db, model = FakeDB(), use_model(FakeModel(), backend="pytorch")
    job = seed_job(db, kind="batch", lines=["Một.", "Hai.", "Ba."], preset="Adam")
    db.tables["vieneu_lines"][1]["voice_preset"] = "Xuân Vĩnh"
    V.speech_job(db, job)

    # Two groups: Adam gets a batch of two, the single Xuân Vĩnh line goes
    # through the per-line path rather than a batch of one.
    assert len(model.batches) == 1, model.batches
    assert model.batches[0][:2] == (["Một.", "Ba."], "Adam"), model.batches
    assert ("Hai.", "Xuân Vĩnh") in model.spoken, model.spoken


@check
def a_failed_batch_falls_back_to_one_line_at_a_time():
    """A batch fails as a unit, so one bad line would take the group with it."""
    db, model = FakeDB(), use_model(FakeModel(fail_on=["Hai."]), backend="pytorch")
    job = seed_job(db, kind="batch", lines=["Một.", "Hai.", "Ba."])
    V.speech_job(db, job)

    assert len(model.batches) == 1, "the batch should be tried once, then dropped"
    row = db.tables["vieneu_jobs"][0]
    assert row["status"] == "done", row
    assert row["qc_summary"]["failed"] == 1, row["qc_summary"]
    lines = {r["idx"]: r["status"] for r in db.tables["vieneu_lines"]}
    assert lines == {0: "done", 1: "failed", 2: "done"}, lines


@check
def a_batch_that_dies_part_way_does_not_re_render_what_it_banked():
    """The fallback must not double-count, or pay twice for audio on disk."""
    db = FakeDB()
    model = use_model(FakeModel(save_fails_after=2), backend="pytorch")
    job = seed_job(db, kind="batch", lines=["Một.", "Hai.", "Ba.", "Bốn."])
    V.speech_job(db, job)

    row = db.tables["vieneu_jobs"][0]
    assert row["status"] == "done", row
    assert row["done_lines"] == 4, row["done_lines"]
    assert row["done_lines"] <= row["total_lines"], row
    # Two banked by the batch, two by the fallback — never one of them twice.
    # Filtered to the per-line keys, since the joined file is a .wav too.
    saved = [p for p in db.uploads if re.search(r"/\d{4}\.wav$", p)]
    assert len(saved) == len(set(saved)) == 4, saved


@check
def a_bad_per_line_voice_fails_only_that_line():
    db, model = FakeDB(), use_model(FakeModel())
    job = seed_job(db, kind="batch", lines=["Một.", "Hai.", "Ba."], preset="Adam")
    db.tables["vieneu_lines"][1]["voice_preset"] = "Nobody"
    V.speech_job(db, job)

    row = db.tables["vieneu_jobs"][0]
    assert row["status"] == "done", row
    lines = {r["idx"]: r["status"] for r in db.tables["vieneu_lines"]}
    assert lines == {0: "done", 1: "failed", 2: "done"}, lines
    assert "Nobody" in (db.tables["vieneu_lines"][1]["error"] or "")


@check
def done_lines_never_exceeds_total_lines():
    # The counter drives a progress bar; a count above the total renders as a
    # bar past 100%, which reads as a bug in the render rather than the count.
    db, model = FakeDB(), use_model(FakeModel(fail_on=["Hai."]))
    job = seed_job(db, lines=["Một.", "Hai.", "Ba."])
    V.speech_job(db, job)
    row = db.tables["vieneu_jobs"][0]
    assert row["done_lines"] <= row["total_lines"], row
    assert row["done_lines"] == 2, row["done_lines"]


@check
def settings_reach_the_model_without_a_worker_change():
    """`settings` is the open contract — a knob added to the model needs no
    deploy here, only a key in the job."""
    db, model = FakeDB(), use_model(FakeModel())
    job = seed_job(db, lines=["Một."], settings={
        "temperature": 0.65, "top_k": 12, "apply_watermark": False,
        # Not model knobs: these belong to the worker and must not be forwarded,
        # or `infer` would reject the call.
        "join_gap_ms": 100, "format": "wav", "backend": "onnx",
    })
    V.speech_job(db, job)
    got = model.gen_kwargs[0]
    assert got == {"temperature": 0.65, "top_k": 12, "apply_watermark": False}, got


@check
def an_absent_setting_is_left_to_the_models_own_default():
    # Sending temperature=None would override the model's 0.8 with nothing.
    db, model = FakeDB(), use_model(FakeModel())
    job = seed_job(db, lines=["Một."], settings={"temperature": None})
    V.speech_job(db, job)
    assert model.gen_kwargs[0] == {}, model.gen_kwargs[0]


# -------------------------------------------------------------------- voices

@check
def a_cloned_voice_is_enrolled_once_and_reused():
    db, model = FakeDB(), use_model(FakeModel())
    db.table("vieneu_voices").insert({
        "id": "v-1", "owner_id": "owner-1", "name": "Chị Trân",
        "sample_path": "owner-1/ref.wav", "denoise": True,
        "consent_confirmed": True, "status": "ready",
    }).execute()
    db.files[V.BUCKET_IN]["owner-1/ref.wav"] = _tone_bytes(4.0)

    job = seed_job(db, lines=["Một.", "Hai.", "Ba."], preset=None,
                   voice_id="v-1")
    V.speech_job(db, job)
    # Three lines, one enrolment — the reference is encoded once per session,
    # not once per line.
    assert len(model.enrolled) == 1, model.enrolled
    assert model.enrolled[0][0] == V.enrol_key("v-1")
    assert all(voice == V.enrol_key("v-1") for _, voice in model.spoken)


@check
def a_cloned_voice_without_consent_is_refused():
    db, model = FakeDB(), use_model(FakeModel())
    db.table("vieneu_voices").insert({
        "id": "v-2", "owner_id": "owner-1", "name": "Ai đó",
        "sample_path": "owner-1/ref.wav", "denoise": True,
        "consent_confirmed": False, "status": "ready",
    }).execute()
    db.files[V.BUCKET_IN]["owner-1/ref.wav"] = _tone_bytes(4.0)

    job = seed_job(db, lines=["Một."], preset=None, voice_id="v-2")
    try:
        V.speech_job(db, job)
    except RuntimeError as e:
        assert "consent" in str(e).lower(), e
    else:
        raise AssertionError("consent must be enforced on the worker too")
    assert model.spoken == []


@check
def a_deleted_cloned_voice_is_reported_not_swallowed():
    db, model = FakeDB(), use_model(FakeModel())
    job = seed_job(db, lines=["Một."], preset=None, voice_id="gone")
    try:
        V.speech_job(db, job)
    except RuntimeError as e:
        assert "deleted" in str(e).lower(), e
    else:
        raise AssertionError("a missing voice row must fail the job by name")


@check
def building_a_voice_writes_an_audition_and_qc():
    db, model = FakeDB(), use_model(FakeModel())
    db.table("vieneu_voices").insert({
        "id": "v-3", "owner_id": "owner-1", "name": "Chị Trân",
        "sample_path": "owner-1/ref.wav", "denoise": True,
        "consent_confirmed": True, "status": "building",
    }).execute()
    db.files[V.BUCKET_IN]["owner-1/ref.wav"] = _tone_bytes(5.0)

    V.build_voice(db, db.tables["vieneu_voices"][0])
    row = db.tables["vieneu_voices"][0]
    assert row["status"] == "ready", row
    assert row["preview_path"] in db.files[V.BUCKET_IN]
    assert abs(row["qc"]["duration_s"] - 5.0) < 0.05, row["qc"]
    assert not row["qc"].get("warnings"), row["qc"]


@check
def a_too_short_reference_is_warned_about_not_rejected():
    # The model will still clone from it. Refusing would be a judgement the
    # worker is not entitled to make; saying so is.
    db, model = FakeDB(), use_model(FakeModel())
    db.table("vieneu_voices").insert({
        "id": "v-4", "owner_id": "owner-1", "name": "Ngắn",
        "sample_path": "owner-1/short.wav", "denoise": True,
        "consent_confirmed": True, "status": "building",
    }).execute()
    db.files[V.BUCKET_IN]["owner-1/short.wav"] = _tone_bytes(1.2)

    V.build_voice(db, db.tables["vieneu_voices"][0])
    row = db.tables["vieneu_voices"][0]
    assert row["status"] == "ready", row
    assert any("under 3s" in w for w in row["qc"]["warnings"]), row["qc"]


@check
def a_voice_with_no_reference_fails_with_a_reason():
    db, model = FakeDB(), use_model(FakeModel())
    db.table("vieneu_voices").insert({
        "id": "v-5", "owner_id": "owner-1", "name": "Trống",
        "sample_path": None, "denoise": True, "consent_confirmed": True,
        "status": "building",
    }).execute()
    try:
        V.build_voice(db, db.tables["vieneu_voices"][0])
    except RuntimeError as e:
        assert "reference" in str(e).lower(), e
    else:
        raise AssertionError("a voice with no clip cannot become ready")


@check
def denoise_off_is_honoured_when_the_build_supports_it():
    db, model = FakeDB(), use_model(FakeModel())
    db.table("vieneu_voices").insert({
        "id": "v-6", "owner_id": "owner-1", "name": "Sạch",
        "sample_path": "owner-1/clean.wav", "denoise": False,
        "consent_confirmed": True, "status": "building",
    }).execute()
    db.files[V.BUCKET_IN]["owner-1/clean.wav"] = _tone_bytes(4.0)
    V.build_voice(db, db.tables["vieneu_voices"][0])
    assert model.enrolled[0][2] == {"denoise": False}, model.enrolled


# ------------------------------------------------------------------ denoise

@check
def denoise_job_uploads_a_clean_file_and_a_level_delta():
    db, model = FakeDB(), use_model(FakeModel())
    db.table("vieneu_jobs").insert({
        "id": "job-d", "owner_id": "owner-1", "kind": "denoise",
        "title": "noisy.wav", "source_path": "owner-1/noisy.wav",
        "settings": {}, "status": "running", "total_lines": 0, "done_lines": 0,
    }).execute()
    db.files[V.BUCKET_IN]["owner-1/noisy.wav"] = _tone_bytes(2.0)

    V.denoise_job(db, db.tables["vieneu_jobs"][0])
    row = db.tables["vieneu_jobs"][0]
    assert row["status"] == "done", row
    assert row["audio_path"] in db.files[V.BUCKET_OUT]
    # The fake quarters the amplitude, which is 12 dB.
    assert abs(row["qc_summary"]["noise_removed_db"] - 12.0) < 0.5, row["qc_summary"]
    assert abs(row["duration_ms"] - 2000) < 50


@check
def denoise_with_no_file_fails_by_name():
    db, model = FakeDB(), use_model(FakeModel())
    db.table("vieneu_jobs").insert({
        "id": "job-e", "owner_id": "owner-1", "kind": "denoise",
        "title": "x", "source_path": None, "settings": {}, "status": "running",
    }).execute()
    try:
        V.denoise_job(db, db.tables["vieneu_jobs"][0])
    except RuntimeError as e:
        assert "upload" in str(e).lower(), e
    else:
        raise AssertionError("a denoise job with nothing to clean cannot succeed")


# ------------------------------------------------------------------ claiming

@check
def claiming_is_atomic_on_the_status_filter():
    db, model = FakeDB(), use_model(FakeModel())
    db.table("vieneu_jobs").insert({
        "id": "job-q", "owner_id": "owner-1", "kind": "speak", "title": "T",
        "settings": {}, "status": "queued", "priority": 5,
    }).execute()

    first = V.claim_next_job(db)
    assert first is not None and first["id"] == "job-q"
    assert db.tables["vieneu_jobs"][0]["status"] == "running"
    # A second worker arriving a moment later finds nothing queued, rather than
    # the same job.
    assert V.claim_next_job(db) is None


def _tone_bytes(seconds: float) -> bytes:
    import io
    n = int(SR * seconds)
    t = np.arange(n, dtype=np.float32) / SR
    tone = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, tone, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def run():
    failed = 0
    for name, fn in checks:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}\n          {e}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"  ERROR {name}\n          {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(checks) - failed}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run())
