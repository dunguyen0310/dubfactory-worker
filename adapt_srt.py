"""Length-aware SRT adaptation ("adaptation" in dubbing-studio terms).

The dubbing pipeline's worst failure mode is an overstuffed cue: forcing a
clip shorter than its words need makes the model drop words, and fit mode has
been measured compressing a cue 1.96x to hit its timestamp. Both are symptoms
of the same root cause — the translated line simply has more syllables than
its time slot can hold at natural pace (~6 syllables/second, the same constant
srt_dub.py uses to estimate durations).

This tool fixes the text instead of the audio. For each cue it computes a
syllable budget (slot seconds x 6), and asks an LLM to rewrite only the lines
that exceed it — same meaning, fewer syllables, natural spoken Vietnamese (or
any language). Digits are spelled out as words at the same time, since the
local install has no text normalization and digits are misread.

    python adapt_srt.py --srt episode.srt --dry-run          # fit report, no API
    python adapt_srt.py --srt episode.srt --output adapted.srt
    python adapt_srt.py --list-models                        # what your key can use

The rewrite step runs on Gemini or Claude, whichever key is present:

    GEMINI_API_KEY (or GOOGLE_API_KEY)  -> Gemini    pip install google-genai
    ANTHROPIC_API_KEY                   -> Claude    pip install anthropic

Both are set with `--provider` if you have both keys. `--dry-run` needs neither:
the fit report — which lines overrun and by how much — is computed locally.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import srt_dub as S

DIGIT_RE = re.compile(r"\d")

# A line is "overstuffed" when it needs more than its slot even after the
# model's usual pacing wiggle room. 1.0 = strict; the default leaves a little
# slack because assemble() also allows a short spill into the next gap.
DEFAULT_TOLERANCE = 1.10

BATCH = 25          # cues rewritten per API call — keeps responses small
MAX_PASSES = 2      # rewrite, then one firmer retry for lines still over

# The shape both providers are asked to return. Gemini's response_schema is an
# OpenAPI subset that rejects `additionalProperties`, so the strict Anthropic
# schema adds it on top rather than the two being maintained separately.
BASE_SCHEMA = {
    "type": "object",
    "properties": {
        "cues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["index", "text"],
            },
        }
    },
    "required": ["cues"],
}


def _strict(node):
    """Anthropic's structured outputs require additionalProperties: false."""
    if isinstance(node, dict):
        out = {k: _strict(v) for k, v in node.items()}
        if out.get("type") == "object":
            out["additionalProperties"] = False
        return out
    if isinstance(node, list):
        return [_strict(v) for v in node]
    return node


SCHEMA = _strict(BASE_SCHEMA)

SYSTEM = """You are a professional dubbing-script adapter. You rewrite \
subtitle lines so they can be spoken naturally within their time slot, the way \
a human dubbing studio adapts a translation.

Rules, in priority order:
1. Keep the meaning, tone, and any brand or product names intact.
2. Fit the syllable budget you are given for each line. Cut filler words, \
prefer shorter synonyms, restructure the sentence — do not cut information \
unless nothing else fits, and never invent new information.
3. Write for the ear, not the eye: natural spoken {language}, no abbreviations.
4. Spell every number, date, and unit as words in {language} (e.g. Vietnamese \
"thứ 8" -> "thứ tám", "50%" -> "năm mươi phần trăm") — the TTS engine misreads \
digits.
5. A line already within budget that contains no digits must be returned \
completely unchanged.

Count syllables the way the pipeline does: one syllable per whitespace-separated \
word that contains a letter or digit (Vietnamese is monosyllabic, so this is \
exact there; treat the budget as a hard word count for other languages).

Return every line you were given, by index."""


def syllable_budget(cue, syl_per_sec: float) -> int:
    return max(2, int(cue.slot * syl_per_sec))


def analyse(cues, syl_per_sec: float, tolerance: float):
    """Return per-cue fit rows and the set of cue indexes needing a rewrite."""
    rows, targets = [], set()
    for c in cues:
        need = S.syllables(c.text)
        budget = syllable_budget(c, syl_per_sec)
        ratio = need / budget if budget else 0.0
        digits = bool(DIGIT_RE.search(c.text))
        over = ratio > tolerance
        if over or digits:
            targets.add(c.index)
        rows.append({
            "index": c.index, "slot": c.slot, "syllables": need,
            "budget": budget, "ratio": ratio, "digits": digits, "over": over,
        })
    return rows, targets


def print_report(rows, title="Fit report"):
    over = [r for r in rows if r["over"]]
    digits = [r for r in rows if r["digits"]]
    print(f"\n{title}: {len(rows)} cues, {len(over)} over budget, "
          f"{len(digits)} with digits")
    if not (over or digits):
        print("  every line fits its slot at natural pace — nothing to adapt")
        return
    print(f"  {'cue':>4}  {'slot':>6}  {'syl':>4}/{'budget':<6} {'ratio':>6}  flags")
    for r in rows:
        if not (r["over"] or r["digits"]):
            continue
        flags = ", ".join(f for f, on in
                          (("OVER", r["over"]), ("digits", r["digits"])) if on)
        print(f"  {r['index']:>4}  {r['slot']:>5.2f}s  {r['syllables']:>4}/"
              f"{r['budget']:<6} {r['ratio']:>5.2f}x  {flags}")


class NoCredentials(RuntimeError):
    """No API key is configured — the caller decides whether that's fatal."""


class BadModel(RuntimeError):
    """The requested model name is not available to this key."""


# ------------------------------------------------------------------ providers
#
# Only the API call itself differs between providers; the syllable budgeting,
# batching, retry and best-of-passes logic above and below is shared. Each
# provider exposes complete(system, prompt) -> JSON text and models() -> list.

GEMINI_KEYS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


class Gemini:
    name = "gemini"
    # The floating alias, deliberately: Gemini's line-up moves fast enough that
    # a pinned id goes stale within months, and this task — short, tightly
    # specified rewriting — is not sensitive to which flash generation runs it.
    # Pin with --model / settings.adapt_model when you need reproducibility.
    default_model = "gemini-flash-latest"
    env = GEMINI_KEYS

    def __init__(self, model=None):
        key = next((os.environ[k] for k in GEMINI_KEYS if os.environ.get(k)), None)
        if not key:
            raise NoCredentials(
                "No Gemini API key. Set GEMINI_API_KEY (get one at "
                "https://aistudio.google.com/apikey).")
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise NoCredentials(
                "The Gemini path needs the Google GenAI SDK:\n"
                "    pip install google-genai") from e
        self._types = types
        self._client = genai.Client(api_key=key)
        self.model = model or self.default_model

    def complete(self, system: str, prompt: str) -> str:
        t = self._types
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=t.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=BASE_SCHEMA,
                    # We pass no tools, and leaving AFC on makes the SDK print a
                    # "not recommended" warning into every worker log line.
                    automatic_function_calling=(
                        t.AutomaticFunctionCallingConfig(disable=True)),
                ),
            )
        except Exception as e:
            msg = str(e)
            # A rejected key is a credentials problem, not a rewrite failure —
            # report it as one so the log says what to fix instead of echoing
            # a raw 400 JSON blob.
            if "API key not valid" in msg or "API_KEY_INVALID" in msg:
                raise NoCredentials(
                    "Gemini rejected the API key. Check GEMINI_API_KEY.") from e
            if "PERMISSION_DENIED" in msg or "UNAUTHENTICATED" in msg:
                raise NoCredentials(
                    "Gemini denied this key access. Check that the key is "
                    "enabled for the Generative Language API.") from e
            # A wrong model name is the other failure worth naming: model
            # line-ups move faster than this script does, so point the user at
            # what their key can actually see instead of a raw 404.
            if "NOT_FOUND" in msg or "not found" in msg.lower():
                raise BadModel(
                    f"Gemini has no model {self.model!r} for this key. "
                    f"Run --list-models to see what is available.") from e
            raise
        text = response.text
        if not text:
            raise RuntimeError(
                "Gemini returned no text — the batch was most likely blocked "
                "by a safety filter. Run --dry-run and adapt those lines by hand.")
        return text

    def models(self):
        return sorted(
            m.name.removeprefix("models/") for m in self._client.models.list()
            if "generateContent" in (getattr(m, "supported_actions", None) or
                                     ["generateContent"]))


class Claude:
    name = "anthropic"
    default_model = "claude-opus-5"
    env = ("ANTHROPIC_API_KEY",)

    def __init__(self, model=None):
        try:
            import anthropic
        except ImportError as e:
            raise NoCredentials(
                "The Claude path needs the Anthropic SDK:\n"
                "    pip install anthropic") from e
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self.model = model or self.default_model

    def complete(self, system: str, prompt: str) -> str:
        a = self._anthropic
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system,
                output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": prompt}],
            )
        except a.NotFoundError as e:
            raise BadModel(f"No Anthropic model {self.model!r}.") from e
        except (a.AuthenticationError, TypeError) as e:
            # The SDK raises TypeError when no credential source exists at all,
            # AuthenticationError when a key exists but is rejected.
            if isinstance(e, TypeError) and "authentication" not in str(e).lower():
                raise
            raise NoCredentials(
                "No usable Anthropic API key. Set ANTHROPIC_API_KEY "
                "(or `ant auth login`).") from e
        if response.stop_reason == "refusal":
            raise RuntimeError("the model declined this batch; run --dry-run "
                               "and adapt those lines by hand")
        return next(b.text for b in response.content if b.type == "text")

    def models(self):
        return sorted(m.id for m in self._client.models.list())


PROVIDERS = {p.name: p for p in (Gemini, Claude)}


def make_client(provider="auto", model=None):
    """Build a rewrite client. 'auto' picks whichever key is present."""
    if provider == "auto":
        for cls in (Gemini, Claude):
            if any(os.environ.get(k) for k in cls.env):
                provider = cls.name
                break
        else:
            raise NoCredentials(
                "No API key found for the rewrite step. Set one of:\n"
                "    GEMINI_API_KEY     (https://aistudio.google.com/apikey)\n"
                "    ANTHROPIC_API_KEY  (https://console.anthropic.com)")
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; "
                         f"pick from {', '.join(PROVIDERS)} or 'auto'")
    return PROVIDERS[provider](model)


def rewrite_batch(client, language, batch, firm=False):
    """Ask the model to rewrite one batch of cues. Returns {index: new_text}."""
    lines = []
    for c, budget, need in batch:
        lines.append(f"[{c.index}] budget {budget} syllables "
                     f"(currently {need}): {c.text}")
    firmness = ("\nThese lines FAILED a previous pass and are still over "
                "budget. Cut harder: drop secondary details if needed. "
                "The budget is a hard limit.") if firm else ""
    prompt = (f"Adapt these dubbing lines to their syllable budgets.{firmness}\n\n"
              + "\n".join(lines))

    text = client.complete(SYSTEM.replace("{language}", language), prompt)
    payload = json.loads(text)
    return {c["index"]: c["text"].strip() for c in payload["cues"]
            if str(c.get("text", "")).strip()}


def adapt(cues, targets, *, language, provider="auto", model=None,
          syllables_per_sec, log=print, tick=None):
    """Rewrite the targeted cues to fit their syllable budgets.

    Importable entry point — worker.py calls this as a pre-render step.
    Returns {cue index: new text}, containing only lines that actually changed.

    `tick`, if given, is called after every API batch. A long episode is many
    batches of slow LLM calls, and the worker uses the tick to heartbeat so
    the stall sweep does not mistake minutes of adaptation for a dead worker.
    """
    client = make_client(provider, model)
    by_index = {c.index: c for c in cues}
    new_text: dict[int, str] = {}

    todo = sorted(targets)
    for attempt in range(1, MAX_PASSES + 1):
        if not todo:
            break
        log(f"pass {attempt}: rewriting {len(todo)} line(s) "
            f"with {client.name}/{client.model}")
        still = []
        for i in range(0, len(todo), BATCH):
            chunk = []
            for idx in todo[i:i + BATCH]:
                c = by_index[idx]
                cur = new_text.get(idx, c.text)
                probe = type(c)(c.index, c.start, c.end, cur)
                chunk.append((probe, syllable_budget(c, syllables_per_sec),
                              S.syllables(cur)))
            out = rewrite_batch(client, language, chunk, firm=attempt > 1)
            if tick:
                tick()
            for probe, budget, _ in chunk:
                idx = probe.index
                candidate = out.get(idx)
                if not candidate:
                    still.append(idx)
                    continue
                # keep whichever wording is closest to budget so a bad pass
                # can never make a line worse
                old = new_text.get(idx, by_index[idx].text)
                best = min((candidate, old),
                           key=lambda t: max(0, S.syllables(t) - budget))
                new_text[idx] = best
                if S.syllables(best) > budget:
                    still.append(idx)
        todo = still
    if todo:
        log(f"  {len(todo)} line(s) still over budget after {MAX_PASSES} "
            f"passes: {todo} — kept the shortest rewrite; consider trimming "
            "by hand or using Even-pacing fit mode.")
    return {i: t for i, t in new_text.items() if t != by_index[i].text}


def write_srt(cues, new_text, path):
    with open(path, "w", encoding="utf-8") as f:
        for n, c in enumerate(cues, 1):
            f.write(f"{n}\n{S.fmt_ts(c.start)} --> {S.fmt_ts(c.end)}\n"
                    f"{new_text.get(c.index, c.text)}\n\n")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--srt", help="subtitle file to adapt")
    p.add_argument("--output", help="adapted .srt (default: <input>.adapted.srt)")
    p.add_argument("--language", default="Vietnamese")
    p.add_argument("--provider", default="auto",
                   choices=["auto", *PROVIDERS],
                   help="auto picks whichever API key is set (Gemini first)")
    p.add_argument("--model", default=None,
                   help="model id; defaults to the provider's "
                        f"({Gemini.default_model} / {Claude.default_model})")
    p.add_argument("--list-models", action="store_true",
                   help="list the models your key can use, then exit")
    p.add_argument("--syllables-per-sec", type=float, default=S.SYLLABLES_PER_SEC,
                   help="speech rate the budget is computed from (default: "
                        "the same 6.0 srt_dub.py uses)")
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                   help="rewrite a line when syllables/budget exceeds this")
    p.add_argument("--dry-run", action="store_true",
                   help="print the fit report and exit — no API call")
    args = p.parse_args()

    if args.list_models:
        try:
            client = make_client(args.provider, args.model)
        except NoCredentials as e:
            sys.exit(str(e))
        print(f"models available to your {client.name} key:")
        for m in client.models():
            print(f"  {m}{'   <- default' if m == client.default_model else ''}")
        return

    if not args.srt:
        p.error("--srt is required (or use --list-models)")

    cues = S.parse_srt(args.srt)
    if not cues:
        sys.exit(f"no cues parsed from {args.srt}")
    print(f"parsed {len(cues)} cues, span {cues[0].start:.1f}s -> "
          f"{cues[-1].end:.1f}s")

    rows, targets = analyse(cues, args.syllables_per_sec, args.tolerance)
    print_report(rows)
    if args.dry_run or not targets:
        return

    try:
        new_text = adapt(cues, targets, language=args.language,
                         provider=args.provider, model=args.model,
                         syllables_per_sec=args.syllables_per_sec)
    except NoCredentials as e:
        sys.exit(f"{e}\n\n(--dry-run gives the fit report with no key at all.)")
    except BadModel as e:
        sys.exit(str(e))

    # after: re-measure with the adapted text
    adapted = [type(c)(c.index, c.start, c.end, new_text.get(c.index, c.text))
               for c in cues]
    rows_after, _ = analyse(adapted, args.syllables_per_sec, args.tolerance)
    print_report(rows_after, title="After adaptation")

    changed = [c.index for c in cues if new_text.get(c.index, c.text) != c.text]
    print(f"\n{len(changed)} line(s) changed: {changed}")
    for idx in changed:
        old = next(c.text for c in cues if c.index == idx)
        print(f"  [{idx}] {old}\n      -> {new_text[idx]}")

    out = args.output or str(Path(args.srt).with_suffix(".adapted.srt"))
    write_srt(cues, new_text, out)
    print(f"\nwrote {out} (timings unchanged — feed it to srt_dub.py / the UI)")


if __name__ == "__main__":
    main()
