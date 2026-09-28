"""Unit checks for hardsub_ocr.py's language handling — no model download.

The OCR engine is replaced by a recorder, so what is checked is what the
module asks of RapidOCR and what it keeps of the answer.

    python test_hardsub_ocr.py
"""

import numpy as np

import hardsub_ocr as H
import translate_srt_deepseek as D

checks = []


def check(name):
    def wrap(fn):
        checks.append((name, fn))
        return fn
    return wrap


class Result:
    def __init__(self, txts):
        self.boxes = [np.array([[10, 10], [200, 10], [200, 50], [10, 50]], np.float32)] * len(txts)
        self.txts = txts
        self.scores = [0.95] * len(txts)


@check("the angle classifier is never used: it turns Korean lines upside down")
def _():
    calls = []

    def engine(img, **kw):
        calls.append(kw)
        return Result(["내가 말했잖아"])

    ocr = object.__new__(H.Ocr)
    ocr.engine, ocr.unclip = engine, 2.2
    boxes = ocr(np.zeros((100, 300, 3), np.uint8), 0.5)
    assert [b["txt"] for b in boxes] == ["내가 말했잖아"], boxes
    assert calls and calls[0]["use_cls"] is False, calls


@check("one-syllable Korean and Chinese lines survive; a lone Latin letter does not")
def _():
    def line(text, conf, dur=800):
        ln = H.Line(start_ms=1000, end_ms=1000 + dur, text=text)
        ln.confidence = conf
        return ln
    kept = H.filter_lines([line("네.", 0.89), line("好。", 0.87), line("嗯", 0.80),
                           line("A", 0.90, 1500), line("응", 0.96, 300)], 250)
    assert [ln.text for ln in kept] == ["네.", "好。"], [ln.text for ln in kept]


@check("DeepSeek renders names by source language, and knows Korean by name")
def _():
    ko = D.SYSTEM_PROMPT.format(source_name="Korean", language="Vietnamese",
                                name_rule=D.name_rule("Korean"), style="")
    zh = D.SYSTEM_PROMPT.format(source_name="Chinese", language="Vietnamese",
                                name_rule=D.name_rule("Chinese"), style="")
    assert "Kim Ji-won" in ko and "Sino-Vietnamese" not in ko, ko
    assert "Sino-Vietnamese" in zh and "Kim Ji-won" not in zh, zh
    assert D.SOURCE_NAMES["ko"] == "Korean"
    gl = D.GLOSSARY_PROMPT.format(source_name="Korean", name_rule=D.name_rule("Korean"))
    assert "Korean short drama" in gl and "romanization" in gl, gl


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
