"""Tests for the cue shaper in transcribe_video.py.

Only `shape_cues` and its helpers are tested here, and deliberately so: the
WhisperX call is a thin wrapper around someone else's model, while the shaper is
where the real edge cases live — words the aligner could not place, segments it
could not align at all, hallucinated runs over music, and the arithmetic that
must never emit an overlapping or negative-length cue.

Every case below is one that real ASR output produces. No GPU, no model
download, no whisperx import:

    python test_transcribe.py
"""

import transcribe_video as T


def words(spec, score=0.9):
    """Build a word list from (text, start, end) triples.

    `None` for a time means the aligner could not place that word — the shape
    real output takes, with the key simply absent rather than null.
    """
    out = []
    for text, start, end in spec:
        w = {"word": text}
        if start is not None:
            w["start"] = start
        if end is not None:
            w["end"] = end
        if score is not None:
            w["score"] = score
        out.append(w)
    return out


def segment(spec, start=None, end=None, score=0.9, text=None):
    ws = words(spec, score) if spec else []
    lo = start if start is not None else (spec[0][1] if spec else 0.0)
    hi = end if end is not None else (spec[-1][2] if spec else 0.0)
    return {
        "start": lo, "end": hi, "words": ws,
        "text": text if text is not None else " ".join(t for t, _, _ in spec),
    }


def evenly(sentence, start, per_word=0.3, score=0.9):
    """A sentence spoken at a steady pace from `start`."""
    spec, t = [], start
    for w in sentence.split():
        spec.append((w, round(t, 3), round(t + per_word, 3)))
        t += per_word
    return segment(spec, score=score)


checks = []


def check(name):
    def wrap(fn):
        checks.append((name, fn))
        return fn
    return wrap


def assert_sane(cues, label=""):
    """Invariants that must hold for every cue list, whatever the input.

    These are the failures that would be invisible in a spot check and fatal in
    a player or in the dub: cues out of order, overlapping, zero-length, or
    silently empty.
    """
    for i, c in enumerate(cues):
        assert c["idx"] == i + 1, f"{label}: idx not sequential at {i}: {c}"
        assert c["end_ms"] > c["start_ms"], f"{label}: non-positive cue {c}"
        assert c["start_ms"] >= 0, f"{label}: negative start {c}"
        assert c["text"].strip(), f"{label}: empty text {c}"
        if i:
            prev = cues[i - 1]
            assert c["start_ms"] >= prev["end_ms"], (
                f"{label}: cue {i + 1} overlaps {i}: "
                f"{prev['end_ms']} -> {c['start_ms']}")


@check("a long monologue is split into readable cues")
def _():
    # 40 words at 0.3 s each: 12 s of speech in one Whisper segment.
    seg = evenly(" ".join(f"word{i}" for i in range(40)), 0.0)
    cues = T.shape_cues([seg], language="en")
    assert_sane(cues, "monologue")
    assert len(cues) > 1, "12s of speech should not be one cue"
    for c in cues:
        assert c["end_ms"] - c["start_ms"] <= (T.MAX_CUE_SECONDS + 0.2) * 1000, c
        assert len(c["text"]) <= T.MAX_CUE_CHARS, c
    # No text may be lost in the split.
    joined = " ".join(c["text"] for c in cues).split()
    assert len(joined) == 40, f"lost words: {len(joined)} of 40"


@check("a pause inside a segment becomes a cue boundary")
def _():
    seg = segment([
        ("Hello", 0.0, 0.4), ("there", 0.4, 0.9),
        # 1.5 s of silence — a real clause break
        ("how", 2.4, 2.7), ("are", 2.7, 3.0), ("you", 3.0, 3.4),
    ])
    cues = T.shape_cues([seg], language="en")
    assert_sane(cues, "pause")
    assert len(cues) == 2, f"pause should split: {cues}"
    assert cues[0]["text"] == "Hello there"
    assert cues[1]["text"] == "how are you"


@check("sentence ends split, but slivers are merged back")
def _():
    seg = segment([
        ("This", 0.0, 0.3), ("is", 0.3, 0.6), ("a", 0.6, 0.8),
        ("full", 0.8, 1.1), ("sentence.", 1.1, 1.6),
        ("Oh.", 1.7, 1.9),          # too short to stand alone
    ])
    cues = T.shape_cues([seg], language="en")
    assert_sane(cues, "sliver")
    assert len(cues) == 1, f"the sliver should have merged: {cues}"
    assert cues[0]["text"].endswith("Oh.")


@check("words the aligner could not place still make it into the text")
def _():
    # "2026" and "Zalo" are exactly what alignment drops: a numeral and a name
    # outside the phoneme dictionary.
    seg = segment([
        ("In", 0.0, 0.3), ("2026", None, None), ("we", 0.9, 1.2),
        ("used", 1.2, 1.5), ("Zalo", None, None),
    ], start=0.0, end=2.2)
    cues = T.shape_cues([seg], language="en")
    assert_sane(cues, "unplaced")
    text = " ".join(c["text"] for c in cues)
    assert "2026" in text and "Zalo" in text, f"dropped a word: {text}"


@check("a segment with no words at all is kept whole")
def _():
    # Alignment failed for this line: no words, but the transcript is there.
    seg = segment([], start=5.0, end=8.0, text="Không thể căn chỉnh dòng này")
    cues = T.shape_cues([seg], language="vi")
    assert_sane(cues, "no words")
    assert len(cues) == 1
    assert cues[0]["text"] == "Không thể căn chỉnh dòng này"
    assert cues[0]["start_ms"] < cues[0]["end_ms"]


@check("hallucinated low-score runs are dropped")
def _():
    good = evenly("this part is real speech", 0.0, score=0.95)
    junk = evenly("thanks for watching please subscribe", 3.0, score=0.05)
    cues = T.shape_cues([good, junk], language="en")
    assert_sane(cues, "junk")
    text = " ".join(c["text"] for c in cues)
    assert "real speech" in text
    assert "subscribe" not in text, f"kept hallucination: {text}"


@check("middling scores are flagged for review, not dropped")
def _():
    seg = evenly("this line is a bit uncertain but real", 0.0, score=0.35)
    cues = T.shape_cues([seg], language="en")
    assert_sane(cues, "low conf")
    assert cues, "a 0.35 score is doubt, not junk — it must survive"
    assert all(c["low_confidence"] for c in cues), cues


@check("a very short cue is extended but never into the next one")
def _():
    # Two clipped words, 1.2 s apart: the first must reach MIN_CUE_SECONDS
    # without touching the second.
    segs = [
        segment([("Yes", 0.0, 0.2)]),
        segment([("No", 1.2, 1.4)]),
    ]
    cues = T.shape_cues(segs, language="en")
    assert_sane(cues, "min duration")
    assert len(cues) == 2, cues
    assert cues[0]["end_ms"] - cues[0]["start_ms"] >= 900, cues[0]
    assert cues[0]["end_ms"] <= cues[1]["start_ms"], cues


@check("rapid dialogue across many short segments stays ordered")
def _():
    segs = [evenly(f"line number {i} spoken quickly", i * 2.0) for i in range(12)]
    cues = T.shape_cues(segs, language="en")
    assert_sane(cues, "dialogue")
    assert len(cues) >= 12, f"expected at least one cue per segment: {len(cues)}"


@check("a long silence between segments creates no phantom cue")
def _():
    segs = [evenly("first thing said", 0.0), evenly("much later", 600.0)]
    cues = T.shape_cues(segs, language="en")
    assert_sane(cues, "silence")
    assert len(cues) == 2, cues
    assert cues[1]["start_ms"] > 599_000


@check("languages without spaces are joined without spaces")
def _():
    seg = segment([("这", 0.0, 0.2), ("是", 0.2, 0.4), ("测", 0.4, 0.6),
                   ("试", 0.6, 0.9)])
    cues = T.shape_cues([seg], language="zh")
    assert_sane(cues, "zh")
    assert cues[0]["text"] == "这是测试", cues[0]["text"]


@check("empty input produces no cues rather than an error")
def _():
    assert T.shape_cues([], language="en") == []
    assert T.shape_cues([segment([], 1.0, 2.0, text="")], language="en") == []


@check("language names resolve to the codes whisperx accepts")
def _():
    # The app's picker sends display names; whisperx wants ISO codes. "Chinese"
    # passed through raw cost a real job — after the GPU model was chosen.
    assert T.lang_code("Chinese") == "zh"
    assert T.lang_code("chinese") == "zh"
    assert T.lang_code("Vietnamese") == "vi"
    assert T.lang_code("Korean") == "ko"
    # Codes pass through, including ones outside the map.
    assert T.lang_code("zh") == "zh"
    assert T.lang_code("ZH") == "zh"
    assert T.lang_code("yue") == "yue"
    assert T.lang_code("haw") == "haw"
    # Detection, in all the spellings callers use for it.
    assert T.lang_code(None) is None
    assert T.lang_code("") is None
    assert T.lang_code("auto") is None
    assert T.lang_code("Auto-Detect") is None
    # An unknown NAME is refused with both accepted forms in the message,
    # before any model has loaded.
    try:
        T.lang_code("Klingon")
    except ValueError as e:
        assert "Chinese" in str(e) and "zh" in str(e), e
    else:
        raise AssertionError("an unknown name must be refused, not passed on")


@check("transient provider errors are retried, real ones are not")
def _():
    class Boom(Exception):
        pass

    e503 = Boom("503 UNAVAILABLE high demand")
    e503b = Boom("x"); e503b.status_code = 503
    e429 = Boom("Resource has been exhausted (RESOURCE_EXHAUSTED)")
    bad = ValueError("Expecting value: line 1 column 1")
    assert T._transient(e503) and T._transient(e503b) and T._transient(e429)
    assert not T._transient(bad), "a JSON error must fail fast, not retry"

    old = T.RETRY_DELAYS
    T.RETRY_DELAYS = (0, 0)
    try:
        class Flaky:
            def __init__(self, failures): self.left, self.calls = failures, 0
            def complete(self, system, prompt):
                self.calls += 1
                if self.left:
                    self.left -= 1
                    raise Boom("503 UNAVAILABLE")
                return '{"cues": []}'

        c = Flaky(2)
        assert T._complete_retry(c, "s", "p", log=lambda m: None) == '{"cues": []}'
        assert c.calls == 3, f"expected 2 retries then success, got {c.calls} calls"

        c = Flaky(99)
        try:
            T._complete_retry(c, "s", "p", log=lambda m: None)
        except RuntimeError as e:
            assert "requeue" in str(e), "the failure must say how to recover"
            assert "transcript is already saved" in str(e)
        else:
            raise AssertionError("exhausted retries must raise")

        class Fatal:
            calls = 0
            def complete(self, system, prompt):
                Fatal.calls += 1
                raise ValueError("bad request")
        try:
            T._complete_retry(Fatal(), "s", "p", log=lambda m: None)
        except ValueError:
            assert Fatal.calls == 1, "non-transient errors must not be retried"
        else:
            raise AssertionError("non-transient error must propagate")
    finally:
        T.RETRY_DELAYS = old


@check("a busy default model falls back to a pinned older flash")
def _():
    import json as _json
    import adapt_srt as A

    class Busy:
        name, model = "gemini", "gemini-flash-latest"
        def complete(self, system, prompt):
            raise RuntimeError("503 UNAVAILABLE high demand")

    class Works:
        name, model = "gemini", "gemini-2.5-flash"
        def complete(self, system, prompt):
            idxs = [int(l.split("]")[0][1:]) for l in prompt.splitlines()
                    if l.startswith("[")]
            return _json.dumps({"cues": [{"index": i, "text": f"vi{i}"}
                                         for i in idxs]})

    made = []
    def fake_make(provider, model):
        made.append((provider, model))
        return Busy() if len(made) == 1 else Works()

    old_make, old_delays = A.make_client, T.RETRY_DELAYS
    A.make_client, T.RETRY_DELAYS = fake_make, ()
    try:
        logs = []
        out = T.translate_cues([{"idx": i, "text": f"line {i}"} for i in (1, 2, 3)],
                               language="Vietnamese", log=logs.append)
        assert len(out) == 3, f"fallback should finish the job: {out}"
        # The first fallback tried must be the pinned older flash, not a guess.
        assert made[1] == ("gemini", "gemini-2.5-flash"), made
        assert any("switching translation" in l for l in logs), logs
    finally:
        A.make_client, T.RETRY_DELAYS = old_make, old_delays


@check("a pinned model is never silently replaced")
def _():
    import adapt_srt as A

    class Busy:
        name, model = "gemini", "my-pinned-model"
        def complete(self, system, prompt):
            raise RuntimeError("503 UNAVAILABLE high demand")

    made = []
    def fake_make(provider, model):
        made.append((provider, model))
        return Busy()

    old_make, old_delays = A.make_client, T.RETRY_DELAYS
    A.make_client, T.RETRY_DELAYS = fake_make, ()
    try:
        try:
            T.translate_cues([{"idx": 1, "text": "line"}],
                             language="Vietnamese", model="my-pinned-model",
                             log=lambda m: None)
        except T.ProviderUnavailable:
            pass
        else:
            raise AssertionError("a busy pinned model must fail, not substitute")
        assert made == [("auto", "my-pinned-model")],             f"pinning must prevent any fallback client: {made}"
    finally:
        A.make_client, T.RETRY_DELAYS = old_make, old_delays


@check("translate_cues flushes each batch as it lands")
def _():
    import json as _json
    import adapt_srt as A

    class FakeClient:
        name, model = "fake", "fake-1"
        def complete(self, system, prompt):
            idxs = [int(l.split("]")[0][1:]) for l in prompt.splitlines()
                    if l.startswith("[")]
            return _json.dumps({"cues": [{"index": i, "text": f"vi{i}"}
                                         for i in idxs]})

    old_make = A.make_client
    A.make_client = lambda p, m: FakeClient()
    try:
        cues = [{"idx": i, "text": f"line {i}"} for i in range(1, 60)]
        batches = []
        out = T.translate_cues(cues, language="Vietnamese",
                               log=lambda m: None, flush=batches.append)
        assert len(out) == 59
        # 59 cues at 25/batch = 3 flushes, in order, disjoint, complete.
        assert len(batches) == 3, f"expected 3 flushed batches, got {len(batches)}"
        seen = [i for b in batches for i in b]
        assert sorted(seen) == list(range(1, 60)), "flushes must cover every cue once"
    finally:
        A.make_client = old_make


@check("the srt writer round-trips through the pipeline's own parser")
def _():
    import tempfile
    from pathlib import Path
    import srt_dub as S

    segs = [evenly("first line of the transcript", 0.0),
            evenly("second line of the transcript", 4.0)]
    cues = T.shape_cues(segs, language="en")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "out.srt"
        T.write_srt(cues, path, {1: "dòng một", 2: "dòng hai"})
        back = S.parse_srt(str(path))
        assert len(back) == len(cues), f"{len(back)} parsed vs {len(cues)} written"
        assert back[0].text == "dòng một", back[0].text
        # Timings must survive the ms -> string -> float round trip.
        for written, parsed in zip(cues, back):
            assert abs(parsed.start - written["start_ms"] / 1000) < 0.002
            assert abs(parsed.end - written["end_ms"] / 1000) < 0.002


@check("untranslated lines fall back to the transcript, not to nothing")
def _():
    import tempfile
    from pathlib import Path
    import srt_dub as S

    segs = [evenly("this one gets translated", 0.0),
            evenly("this one does not", 4.0)]
    cues = T.shape_cues(segs, language="en")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "partial.srt"
        T.write_srt(cues, path, {1: "đã dịch"})
        back = S.parse_srt(str(path))
        assert back[0].text == "đã dịch"
        assert "this one does not" in back[-1].text, back[-1].text


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
            print(f"  ERROR {name}\n          {type(e).__name__}: {e}")
    print(f"\n{len(checks) - failed}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
