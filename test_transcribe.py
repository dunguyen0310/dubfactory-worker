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


@check("a mid-segment unplaced word cannot swallow the segment tail")
def _():
    # A(0-1), B unplaced, C(2-3), D unplaced, segment end 10. The old backward
    # pass gave B end = start + (10-3)/2 = 4.5, overlapping C entirely. B must
    # end where C begins; only the trailing run (D) may spread to segment end.
    seg = segment([
        ("A", 0.0, 1.0), ("B", None, None),
        ("C", 2.0, 3.0), ("D", None, None),
    ], start=0.0, end=10.0)
    ws = T._words_with_times(seg)
    assert ws[1]["end"] <= ws[2]["start"] + 1e-6, f"B swallowed the tail: {ws[1]}"
    assert ws[3]["start"] < ws[3]["end"] <= 10.0 + 1e-6, ws[3]
    cues = T.shape_cues([seg], language="en")
    assert_sane(cues, "mid-unplaced")


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


# ------------------------------------------------ Chinese, the way whisperx
# returns it: every character — punctuation included — its own "word", and the
# spaces Whisper writes between unpunctuated phrases dropped from the words
# but still present in the segment text.

ZH_PUNCT = "，。？！、；："


def zh(text, start=0.0, per_char=0.2, space_pause=0.35, score=0.9):
    spec, t = [], start
    for ch in text:
        if ch.isspace():
            t += space_pause
            continue
        dur = 0.08 if ch in ZH_PUNCT else per_char
        spec.append((ch, round(t, 3), round(t + dur, 3)))
        t += dur
    return segment(spec, score=score, text=text)


def texts(cues):
    return [c["text"] for c in cues]


@check("zh: every sentence gets its own cue (the six-second-block bug)")
def _():
    # Three sentences spoken back to back. The Latin-tuned shaper glued them
    # into one cue — a sentence end only cut once a cue held 28 characters,
    # and anything under 12 characters was merged into the line before.
    cues = T.shape_cues([zh("你怎么在这里？我来找你。你不该来的。")], language="zh")
    assert_sane(cues, "zh sentences")
    assert texts(cues) == ["你怎么在这里？", "我来找你。", "你不该来的。"], texts(cues)


@check("zh: a longer scene is cut at clause marks, never across a sentence end")
def _():
    line = "我告诉你，这件事情跟你没有关系。你最好别管！不然的话，我对你不客气。"
    cues = T.shape_cues([zh(line)], language="zh")
    assert_sane(cues, "zh scene")
    assert all(len(c["text"]) <= 16 for c in cues), texts(cues)
    for c in cues:
        # Every cue ends at punctuation, and a sentence end only ever ends one.
        assert c["text"][-1] in "。！？，", f"cut mid-clause: {c['text']}"
        assert not any(p in c["text"][:-1] for p in "。！？"), \
            f"spans a sentence end: {c['text']}"
    assert "".join(texts(cues)) == line, "text was lost or reordered"


@check("zh: unpunctuated speech is cut at the phrase spaces Whisper wrote")
def _():
    cues = T.shape_cues([zh("你怎么在这里 我来找你 你不该来的")], language="zh")
    assert_sane(cues, "zh spaces")
    assert texts(cues) == ["你怎么在这里", "我来找你", "你不该来的"], texts(cues)


@check("zh: a phrase space inside one cue is kept as a space")
def _():
    # A vocative and its sentence read as one subtitle line, the way burned-in
    # Chinese subtitles write it: "曼云 我们走".
    cues = T.shape_cues([zh("曼云 我们走吧", space_pause=0.1)], language="zh")
    assert texts(cues) == ["曼云 我们走吧"], texts(cues)


@check("zh: nothing ever exceeds one 16-character line, even with no cues at all")
def _():
    line = "我们今天晚上一起去看电影然后吃火锅再去唱歌最后回家睡觉明天早上八点起床去上班"
    cues = T.shape_cues([zh(line)], language="zh")
    assert_sane(cues, "zh run-on")
    assert all(len(c["text"]) <= 16 for c in cues), texts(cues)
    assert "".join(texts(cues)) == line


@check("zh: no cue is cut inside a word (with jieba installed)")
def _():
    try:
        import jieba
    except ImportError:
        print("          (skipped: jieba not installed)")
        return
    line = "我们今天晚上一起去看电影然后吃火锅怎么样"
    cues = T.shape_cues([zh(line)], language="zh")
    edges, pos = set(), 0
    for piece in jieba.lcut(line):
        pos += len(piece)
        edges.add(pos)
    pos = 0
    for c in cues[:-1]:
        pos += len(c["text"])
        assert pos in edges, f"cut inside a word at {pos}: {texts(cues)}"


@check("zh: punctuation is never a cue of its own nor split from its word")
def _():
    line = "他说：“走吧。”好！你呢？"
    cues = T.shape_cues([zh(line)], language="zh")
    assert_sane(cues, "zh punct")
    for c in cues:
        assert c["text"][0] not in "，。？！：”", f"starts with punctuation: {c['text']}"
        assert c["text"][-1] != "“", f"ends with an opening quote: {c['text']}"
    assert "".join(texts(cues)) == line


@check("zh: a passage the aligner gave up on is cut at estimated times, flagged")
def _():
    text = ("我告诉你，这件事情跟你没有关系。你最好别管！不然的话，我对你不客气。"
            "你听明白了吗？")
    seg = segment([], start=10.0, end=22.0, text=text)
    cues = T.shape_cues([seg], language="zh")
    assert_sane(cues, "zh unaligned")
    assert len(cues) >= 4, texts(cues)
    assert all(len(c["text"]) <= 16 for c in cues), texts(cues)
    assert all(c["low_confidence"] and "estimated" in (c["note"] or "")
               for c in cues), cues
    assert "".join(texts(cues)) == text
    assert cues[0]["start_ms"] >= 9_800 and cues[-1]["end_ms"] <= 22_500, cues


# ----------------------------------------------------------------- timing

@check("a cue appears just before its first word and lingers after its last")
def _():
    seg = zh("你好，我是小明。", start=1.0)
    speech_end = seg["words"][-2]["end"]          # 明; the 。 after it is silence
    cues = T.shape_cues([seg, zh("再见。", start=6.0)], language="zh")
    assert len(cues) == 2, cues
    assert cues[0]["start_ms"] == 900, cues[0]    # a 0.1 s lead-in
    assert cues[0]["end_ms"] == round((speech_end + T.LAG_OUT_SECONDS) * 1000), cues[0]


@check("a short gap between two lines is closed to two frames")
def _():
    segs = [zh("你好。", start=1.0), zh("再见。", start=2.1)]
    cues = T.shape_cues(segs, language="zh")
    assert len(cues) == 2, cues
    gap = cues[1]["start_ms"] - cues[0]["end_ms"]
    assert gap == round(T.MIN_GAP_SECONDS * 1000), (gap, cues)


@check("a cue never appears before the previous line's speech has ended")
def _():
    # 0.03 s between two sentences: the second may not borrow a lead-in from
    # time the first is still being spoken in.
    seg = zh("你好。", start=1.0)
    seg2 = zh("我是小明。", start=1.43)
    cues = T.shape_cues([seg, seg2], language="zh")
    assert len(cues) == 2, cues
    assert cues[1]["start_ms"] >= 1400, cues             # 你好 ends at 1.4 s
    assert cues[1]["start_ms"] <= 1430, cues             # and not after 我 starts
    assert cues[0]["end_ms"] <= cues[1]["start_ms"]


@check("the audio track's start offset is measured from the container")
def _():
    import json as _json
    import subprocess

    def probe(audio, video):
        streams = [{"codec_type": "video", "start_time": str(video)},
                   {"codec_type": "audio", "start_time": str(audio)}]
        return type("R", (), {"stdout": _json.dumps({"streams": streams})})()

    old_which, old_run = T.shutil.which, T.subprocess.run
    T.shutil.which = lambda name: "/usr/bin/ffprobe"
    try:
        for audio, video, want in ((0.3, 0.0, 0.3), (0.0, 0.1, 0.0),
                                   (0.01, 0.0, 0.0), (0.5, 0.2, 0.3)):
            T.subprocess.run = lambda *a, **k: probe(audio, video)
            got = T.media_offset("x.mp4")
            assert abs(got - want) < 1e-6, (audio, video, got)
        T.shutil.which = lambda name: None
        assert T.media_offset("x.mp4") == 0.0
    finally:
        T.shutil.which, T.subprocess.run = old_which, old_run
    assert subprocess.run is old_run


@check("timings are moved onto the video's timeline, words included")
def _():
    segs = [{"start": 1.0, "end": 2.0, "text": "hi",
             "words": [{"word": "hi", "start": 1.0, "end": 1.5}]}]
    T._shift(segs, 0.25)
    assert segs[0]["start"] == 1.25 and segs[0]["words"][0]["end"] == 1.75, segs


# --------------------------------------------------------------- ASR text

@check("Whisper's credits hallucinations and prompt echoes are dropped")
def _():
    prompt = T.ASR_PROMPTS["zh"]
    segs = [{"text": "字幕由Amara.org社区提供"},
            {"text": "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目"},
            {"text": "以下是普通话的对白"},
            {"text": "你今天去哪里了？我到处找你。"},
            {"text": "我们一起订阅了那本杂志，然后转发给了所有同事，"
                     "大家都说写得很好，我觉得你也应该看看，真的很不错。"}]
    kept = T._drop_junk_segments(segs, prompt, log=lambda m: None)
    assert [s["text"][:4] for s in kept] == ["你今天去", "我们一起"], kept


@check("the ASR prompt: auto by language, custom as given, or off")
def _():
    assert T._asr_prompt("auto", "zh") == T.ASR_PROMPTS["zh"]
    assert T._asr_prompt(None, "zh") == T.ASR_PROMPTS["zh"]
    assert T._asr_prompt("auto", "en") is None
    assert T._asr_prompt("off", "zh") is None and T._asr_prompt(False, "zh") is None
    assert T._asr_prompt("林小雨，陆景深。", "zh") == "林小雨，陆景深。"

    import dataclasses

    @dataclasses.dataclass
    class Options:
        beam_size: int = 5
        initial_prompt: str | None = None

    model = type("M", (), {})()
    model.options = Options()
    assert T._set_initial_prompt(model, "提示。")
    assert model.options.initial_prompt == "提示。" and model.options.beam_size == 5
    assert not T._set_initial_prompt(type("N", (), {})(), "x")


@check("saved word timings re-cut to exactly the same cues")
def _():
    import json as _json
    segs = [zh("你怎么在这里？我来找你。", start=2.0),
            zh("你不该来的。", start=6.0)]
    segs[0]["words"][3]["score"] = float("nan")
    saved = _json.loads(_json.dumps(T.words_json(
        {"language": "zh", "alignment": "word", "segments": segs,
         "duration": 8.0, "offset": 0.0})))
    assert saved["segments"][0]["words"][3].get("score") is None
    a = T.shape_cues(segs, language="zh")
    b = T.shape_cues(saved["segments"], language="zh")
    assert [(c["start_ms"], c["end_ms"], c["text"]) for c in a] == \
           [(c["start_ms"], c["end_ms"], c["text"]) for c in b]


# ------------------------------------------------------------ translation

def _prompt_client(reply):
    """A client that records every prompt and answers with reply(idx, prompt)."""
    import json as _json

    class Client:
        name, model = "fake", "fake-1"
        prompts = []

        def complete(self, system, prompt):
            Client.prompts.append(prompt)
            idxs = [int(l.split("]")[0][1:]) for l in prompt.splitlines()
                    if l.startswith("[")]
            return _json.dumps({"cues": [{"index": i, "text": reply(i, prompt)}
                                         for i in idxs]})
    return Client


@check("each cue goes to the translator with its duration and a length budget")
def _():
    import adapt_srt as A
    Client = _prompt_client(lambda i, p: f"vi{i}")
    old = A.make_client
    A.make_client = lambda p, m: Client()
    try:
        cues = [{"idx": 1, "text": "你好", "start_ms": 0, "end_ms": 2000},
                {"idx": 2, "text": "再见", "start_ms": 3000, "end_ms": 3600}]
        T.translate_cues(cues, language="Vietnamese", log=lambda m: None)
        first = Client.prompts[0]
        assert "[1] (2.0s, max 34 chars) 你好" in first, first
        # 0.6 s x 17 cps is 10 characters, but no line is budgeted below half
        # a display line: a short cue still needs room for a whole thought.
        assert "[2] (0.6s, max 21 chars) 再见" in first, first
        assert T.char_budget(10, "Vietnamese") == 84      # never past two lines
        assert T.char_budget(2, "Chinese") == 18
    finally:
        A.make_client = old


@check("an over-long translation is condensed once, and only a shorter one kept")
def _():
    import adapt_srt as A
    long_vi = "Anh không nên đến đây vào lúc này, ở đây rất nguy hiểm cho anh và cả gia đình anh nữa"

    def reply(i, prompt):
        if prompt.startswith("These Vietnamese subtitle translations are too long"):
            return "Anh không nên đến, ở đây nguy hiểm." if i == 1 else long_vi + " nữa"
        return long_vi
    Client = _prompt_client(reply)
    old = A.make_client
    A.make_client = lambda p, m: Client()
    try:
        flushed = []
        cues = [{"idx": 1, "text": "你不该来的，这里很危险", "start_ms": 0, "end_ms": 2000},
                {"idx": 2, "text": "你不该来的，这里很危险", "start_ms": 3000, "end_ms": 5000}]
        out = T.translate_cues(cues, language="Vietnamese", log=lambda m: None,
                               flush=flushed.append)
        assert out[1] == "Anh không nên đến, ở đây nguy hiểm.", out
        assert out[2] == long_vi, "a longer 'condensed' line must not replace the original"
        assert any(1 in b and b[1] == out[1] for b in flushed), flushed
        assert sum(p.startswith("These Vietnamese") for p in Client.prompts) == 1
    finally:
        A.make_client = old


@check("later batches see earlier cues with the translation chosen for them")
def _():
    import adapt_srt as A
    Client = _prompt_client(lambda i, p: f"vi{i}")
    old = A.make_client
    A.make_client = lambda p, m: Client()
    try:
        cues = [{"idx": i, "text": f"句子{i}"} for i in range(1, 31)]
        T.translate_cues(cues, language="Vietnamese", log=lambda m: None)
        second = Client.prompts[1]
        assert "CONTEXT [25]: 句子25  =>  vi25" in second, second
    finally:
        A.make_client = old


# ------------------------------------------------------------------- srt

@check("a long Vietnamese line is wrapped in two at a natural break")
def _():
    text = "Anh không nên đến đây vào lúc này, ở đây rất nguy hiểm cho anh"
    out = T.wrap_text(text, 42)
    top, bottom = out.split("\n")
    assert len(top) <= 42 and len(bottom) <= 42, out
    assert top.endswith(","), f"should break after the comma: {out!r}"
    assert T.wrap_text("Anh làm gì ở đây?", 42) == "Anh làm gì ở đây?"
    zh_out = T.wrap_text("我告诉你这件事情跟你没有关系，你最好别管这件事", 16, spaced=False)
    assert zh_out.split("\n")[0].endswith("，"), zh_out


@check("a wrapped .srt still parses back to the same text")
def _():
    import tempfile
    from pathlib import Path
    import srt_dub as S

    cues = [{"idx": 1, "start_ms": 0, "end_ms": 3000, "text": "x"}]
    vi = "Anh không nên đến đây vào lúc này, ở đây rất nguy hiểm cho anh"
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "vi.srt"
        T.write_srt(cues, path, {1: vi}, wrap="Vietnamese")
        raw = path.read_text(encoding="utf-8")
        assert raw.count("\n") >= 5, raw           # the text took two lines
        assert S.parse_srt(str(path))[0].text == vi


@check("shape numbers describe the cues")
def _():
    cues = T.shape_cues([zh("你怎么在这里？我来找你。你不该来的。")], language="zh")
    st = T.shape_stats(cues, "zh")
    assert st["profile"] == "han" and st["cue_chars_max"] <= 16, st
    assert st["cues_over_limit"] == 0 and st["cues_estimated"] == 0, st


@check("zh: Whisper's half-width punctuation is set full-width, numbers untouched")
def _():
    cues = T.shape_cues([zh("你好,我是谁?")], language="zh")
    assert texts(cues) == ["你好，我是谁？"], texts(cues)
    assert T._full_width("他花了1,000块。OK?") == "他花了1,000块。OK?"


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
