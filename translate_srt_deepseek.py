#!/usr/bin/env python
"""
translate_srt_deepseek.py — OCR'ed .srt (Chinese / English) -> Vietnamese .srt via the DeepSeek API.

    python translate_srt_deepseek.py ep01.srt                          # -> ep01.vi.srt
    python translate_srt_deepseek.py "D:/srt" --workers 4              # every .srt in the folder, 4 files at a time
    python translate_srt_deepseek.py "D:/srt" --build-glossary          # pass 1: names/terms of the series -> glossary.json
    python translate_srt_deepseek.py "D:/srt" --glossary glossary.json --bilingual

Timings are kept exactly; one cue in -> one cue out, so the .vi.srt drops straight into a
Dub Factory subtitle/video dub job.  Unlike translate_srt.py this needs no Dub Factory
checkout: only `pip install openai` and DEEPSEEK_API_KEY (or --api-key).

What it does per file
  1. clean   — with the .ocr.json sidecar next to the .srt, cues whose OCR confidence is below
               --min-confidence (0.9) are dropped; for Chinese sources, lines without a single
               CJK character (licence plates, logos, "GV-6606", "setY re") are dropped and short
               lower-case/digit fragments glued to a line ("宁可睡地板 it", "os曼云") are stripped.
               Measured on 25 episodes of 凤凰特工: real cues score >= 0.95, junk 0.57-0.86.
               --no-clean keeps everything.
  2. translate — the whole episode goes to DeepSeek in chunks of --chunk cues (60), each chunk
               with the previous chunk's last lines as context, the series glossary and the cue
               durations, as JSON in / JSON out (response_format=json_object, thinking off,
               temperature 1.3 = DeepSeek's recommendation for translation).  Missing or extra
               indices are retried; a cue that still fails keeps its source text and is listed.
  3. write   — <name>.vi.srt (renumbered after cleaning), optionally <name>.zh-vi.srt with both
               languages for review, and one row in translate_report.csv.

Resumable: a file whose .vi.srt exists is skipped unless --force.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("translate")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"      # cheapest V4 model; JSON output + thinking-off supported
DEFAULT_TEMPERATURE = 1.3             # api-docs.deepseek.com/quick_start/parameter_settings: translation 1.3

LANG_SUFFIX = {"vietnamese": "vi", "english": "en", "chinese": "zh", "japanese": "ja", "korean": "ko",
               "thai": "th", "french": "fr", "spanish": "es", "german": "de", "indonesian": "id"}
TIME_RE = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
# junk glued to a real line: <= 3 lower-case letters / digits / accented letters at either end
FRAG_HEAD = re.compile(r"^[a-z0-9\u00e0-\u00ff][a-z0-9.\u00e0-\u00ff]{0,2}\s*(?=[\u4e00-\u9fff])")
FRAG_TAIL = re.compile(r"(?<=[\u4e00-\u9fff])\s*[a-z0-9\u00e0-\u00ff][a-z0-9.\u00e0-\u00ff]{0,2}$")
STROKE_CHARS = set("一二丨ー—–-_=~·•.,:;'\"` 　")   # what a bright edge reads as (same set as hardsub_ocr.py)


def lang_suffix(language: str) -> str:
    return LANG_SUFFIX.get(language.strip().lower(), language.strip().lower()[:2])


# ---------------------------------------------------------------- srt
@dataclass
class Cue:
    index: int            # index in the source .srt (matches .ocr.json "lines[].index")
    start: str
    end: str
    text: str             # lines joined with "\n"
    confidence: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        def ms(t: str) -> int:
            h, m, s = t.split(":")
            s, x = s.split(",")
            return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(x)
        return max(0.1, (ms(self.end) - ms(self.start)) / 1000.0)


def parse_srt(path: str | Path) -> list[Cue]:
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace").replace("\r", "")
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        m = None
        for i, ln in enumerate(lines[:2]):
            m = TIME_RE.search(ln)
            if m:
                text = "\n".join(lines[i + 1:]).strip()
                idx = int(lines[0]) if i == 1 and lines[0].isdigit() else len(cues) + 1
                break
        if not m or not text:
            continue
        g = m.groups()
        start = f"{int(g[0]):02d}:{int(g[1]):02d}:{int(g[2]):02d},{int(g[3]):03d}"
        end = f"{int(g[4]):02d}:{int(g[5]):02d}:{int(g[6]):02d},{int(g[7]):03d}"
        cues.append(Cue(idx, start, end, text))
    return cues


def write_srt(cues: list[Cue], translations: dict[int, str], path: Path, bilingual: bool = False) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for n, c in enumerate(cues, 1):
            vi = translations.get(c.index) or c.text
            body = f"{c.text}\n{vi}" if bilingual else vi
            f.write(f"{n}\n{c.start} --> {c.end}\n{body}\n\n")


def load_sidecar(srt: Path) -> dict[int, float]:
    """{srt index: OCR confidence} from <name>.ocr.json written by hardsub_ocr.py, if present."""
    side = srt.with_suffix(".ocr.json")
    if not side.exists():
        return {}
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
        return {int(l["index"]): float(l["confidence"]) for l in data.get("lines", [])}
    except Exception as e:  # noqa: BLE001 — a bad sidecar must not stop the translation
        log.warning("%s: cannot read sidecar (%s) — no confidence filter", srt.name, e)
        return {}


# ---------------------------------------------------------------- 1. clean
def clean_cues(cues: list[Cue], confidences: dict[int, float], *, min_confidence: float,
               cjk_source: bool) -> tuple[list[Cue], list[Cue]]:
    """Returns (kept, dropped).  Kept cues may have had junk lines/fragments removed (see .notes).

    Order matters: junk lines are removed first, and the confidence rule then only drops a cue
    that had no junk removed (a clean line read at 0.8 is probably just hard to read) or whose
    remainder is a single character — a genuine line next to a licence plate scores ~0.85 as a
    whole, but the line itself is fine once the plate is gone."""
    kept, dropped = [], []
    for c in cues:
        c.confidence = confidences.get(c.index)
        removed = 0
        if cjk_source:
            lines = []
            for ln in c.text.split("\n"):
                if not CJK_RE.search(ln):
                    c.notes.append(f"dropped line {ln!r} (no CJK)")
                    removed += 1
                    continue
                new = FRAG_TAIL.sub("", FRAG_HEAD.sub("", ln)).strip()
                if new != ln:
                    c.notes.append(f"stripped fragment: {ln!r} -> {new!r}")
                if not new or all(ch in STROKE_CHARS for ch in new):
                    c.notes.append(f"dropped line {ln!r} (stroke junk)")
                    removed += 1
                    continue
                lines.append(new)
            c.text = "\n".join(lines)
            if not c.text:
                c.notes.append("nothing left")
                dropped.append(c)
                continue
        cjk_left = len(CJK_RE.findall(c.text)) if cjk_source else 2
        if c.confidence is not None and c.confidence < min_confidence and (removed == 0 or cjk_left < 2):
            c.notes.append(f"confidence {c.confidence:.2f} < {min_confidence}")
            dropped.append(c)
            continue
        kept.append(c)
    return kept, dropped


# ---------------------------------------------------------------- 2. translate
SYSTEM_PROMPT = """You are a professional subtitle translator for the Vietnamese dubbing of {source_name} short dramas (短剧).
Translate every cue from {source_name} into natural, spoken Vietnamese — what the character would actually say when dubbed, not a literal rendering.

Rules
- Reply with JSON only, exactly this shape: {{"cues": [{{"i": 12, "vi": "..."}}, ...]}} — one entry per input cue, same "i" values, same order. Never merge, split, drop or add cues, never leave "vi" empty.
- Each cue has "sec", its on-screen duration. Keep the Vietnamese speakable in that time: about as many Vietnamese syllables as the source has characters, short colloquial phrasing, no padding.
- Keep forms of address (xưng hô: anh/em, tôi/cô, ông/tôi, mày/tao, ...) consistent for each pair of characters across the whole episode; infer the relationship from context and keep it stable. Use the "context" cues (already translated) to stay consistent with what came before.
- Names, nicknames, titles, places and organisations: use the glossary exactly as given; otherwise render Chinese names in Sino-Vietnamese and keep them identical every time.
- Plain text only: no translator notes, no brackets, no quotation marks around a line, no explanations. Keep the sentence-final tone (questions, exclamations) and translate interjections naturally (啊 -> à / hả / ơi as fits).
- If a cue is an OCR fragment that makes no sense, still give the most plausible short line.
{style}"""


class DeepSeek:
    """Thin wrapper over the OpenAI-compatible endpoint: JSON output, thinking off, retries."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE,
                 base_url: str = DEEPSEEK_BASE_URL, timeout: float = 180.0):
        from openai import OpenAI  # imported here so --dry-run works without the package
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        self.model, self.temperature = model, temperature
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self._lock = threading.Lock()

    def json_call(self, system: str, user: str, max_tokens: int, attempts: int = 5) -> dict:
        from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
        last: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    response_format={"type": "json_object"},
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                if r.usage:
                    with self._lock:
                        self.usage["prompt_tokens"] += r.usage.prompt_tokens or 0
                        self.usage["completion_tokens"] += r.usage.completion_tokens or 0
                content = (r.choices[0].message.content or "").strip()
                if not content:                       # documented DeepSeek quirk: occasional empty content
                    raise ValueError("empty response")
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
                return json.loads(content)
            except (RateLimitError, APITimeoutError, APIConnectionError, ValueError, json.JSONDecodeError) as e:
                last = e
            except APIStatusError as e:
                last = e
                if e.status_code < 500 and e.status_code != 429:
                    raise
            wait = min(60.0, 2.0 ** attempt + random.random())
            log.warning("DeepSeek call failed (%s: %s) — retry %d/%d in %.0fs", type(last).__name__,
                        str(last)[:120], attempt, attempts, wait)
            time.sleep(wait)
        raise RuntimeError(f"DeepSeek call failed after {attempts} attempts: {last}")


def _chunks(cues: list[Cue], size: int) -> list[list[Cue]]:
    if len(cues) <= size * 1.3:            # a 70-cue episode is one request, not 60 + 10
        return [cues]
    return [cues[i:i + size] for i in range(0, len(cues), size)]


def translate_cues(cues: list[Cue], client: DeepSeek, *, language: str, source: str, glossary: dict[str, str],
                   style: str, chunk: int, log_prefix: str = "", flush=None) -> tuple[dict[int, str], list[int]]:
    """{index: translation}, plus the indices that could not be translated.

    `flush`, if given, is called with each chunk's {index: translation} as soon as that chunk is
    back from the API — the Dub Factory worker persists per chunk so a run that dies halfway
    keeps what it already paid for. The CLI does not need it: one file is one write."""
    source_name = {"zh": "Chinese", "ch": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean"}.get(source, source)
    system = SYSTEM_PROMPT.format(source_name=source_name, language=language,
                                  style=("Style notes from the producer: " + style) if style else "")
    out: dict[int, str] = {}
    failed: list[int] = []
    context: list[dict] = []
    for ci, part in enumerate(_chunks(cues, chunk), 1):
        payload = {
            "target_language": language,
            "glossary": glossary,
            "context": context[-8:],
            "cues": [{"i": c.index, source: c.text.replace("\n", " "), "sec": round(c.seconds, 1)} for c in part],
        }
        user = ("Translate the \"cues\" below into " + language + ". Return json: "
                "{\"cues\": [{\"i\": <same i>, \"vi\": \"<translation>\"}, ...]}\n\n"
                + json.dumps(payload, ensure_ascii=False))
        wanted = {c.index for c in part}
        got: dict[int, str] = {}
        for attempt in range(1, 4):
            data = client.json_call(system, user, max_tokens=min(16000, 400 + 80 * len(part)))
            for item in data.get("cues", []) if isinstance(data, dict) else []:
                try:
                    i, vi = int(item["i"]), str(item["vi"]).strip()
                except (KeyError, TypeError, ValueError):
                    continue
                if i in wanted and vi:
                    got[i] = vi
            missing = wanted - set(got)
            if not missing:
                break
            log.warning("%schunk %d: %d cue(s) missing (%s) — retry %d", log_prefix, ci, len(missing),
                        sorted(missing)[:8], attempt)
            user = ("Some cues were missing or empty in your previous answer. Translate ALL of these cues into "
                    + language + " and return json {\"cues\": [{\"i\": <same i>, \"vi\": \"...\"}]}:\n\n"
                    + json.dumps({"glossary": glossary, "context": context[-8:],
                                  "cues": [{"i": c.index, source: c.text.replace("\n", " "), "sec": round(c.seconds, 1)}
                                           for c in part if c.index in missing]}, ensure_ascii=False))
            wanted = missing
        else:
            failed.extend(sorted(wanted - set(got)))
        out.update(got)
        if flush and got:
            flush(dict(got))
        context = [{source: c.text.replace("\n", " "), "vi": got.get(c.index, "")} for c in part if c.index in got]
    return out, failed


# ---------------------------------------------------------------- glossary
GLOSSARY_PROMPT = """You prepare a translation glossary for the Vietnamese dubbing of a {source_name} short drama.
From the subtitle lines below, list every recurring proper noun: character names and nicknames, forms of address that work like names (e.g. 老板, 哥), titles, places, organisations, codenames, product or brand names.
For each give ONE fixed Vietnamese rendering: Sino-Vietnamese for Chinese personal names (曼云 -> Mạn Vân, 凤凰 -> Phượng Hoàng), the usual Vietnamese name for real places, the sense in Vietnamese for descriptive codenames.
Reply with JSON only: {{"glossary": {{"<source term>": "<Vietnamese>", ...}}}}.  Omit ordinary words."""


def build_glossary(files: list[Path], client: DeepSeek, *, source: str, max_lines: int = 4000) -> dict[str, str]:
    lines = []
    for f in files:
        lines.extend(c.text for c in parse_srt(f))
    return glossary_from_lines(lines, client, source=source, max_lines=max_lines)


def glossary_from_lines(texts: list[str], client: DeepSeek, *, source: str, max_lines: int = 4000) -> dict[str, str]:
    """The glossary pass over subtitle lines already in memory — what the worker has, having
    inserted the cues rather than written a file. `build_glossary` is this over a folder."""
    seen, lines = set(), []
    for text in texts:
        t = text.replace("\n", " ")
        if t not in seen:
            seen.add(t)
            lines.append(t)
    lines = lines[:max_lines]
    source_name = {"zh": "Chinese", "ch": "Chinese"}.get(source, source)
    data = client.json_call(GLOSSARY_PROMPT.format(source_name=source_name),
                            "Subtitle lines (json list):\n" + json.dumps(lines, ensure_ascii=False),
                            max_tokens=4000)
    gl = data.get("glossary", {}) if isinstance(data, dict) else {}
    return {str(k).strip(): str(v).strip() for k, v in gl.items() if str(k).strip() and str(v).strip()}


# ---------------------------------------------------------------- driver
REPORT_FIELDS = ["file", "output", "status", "cues_in", "dropped", "cues_out", "translated", "kept_source",
                 "prompt_tokens", "completion_tokens", "seconds", "error"]


def translate_file(src: Path, dst: Path, client: DeepSeek | None, *, language: str, source: str,
                   glossary: dict[str, str], style: str, chunk: int, clean: bool, min_confidence: float,
                   bilingual: bool, dry_run: bool) -> dict:
    t0 = time.time()
    row = dict(file=src.name, output=dst.name, status="ok", cues_in=0, dropped=0, cues_out=0, translated=0,
               kept_source=0, prompt_tokens=0, completion_tokens=0, seconds=0, error="")
    cues = parse_srt(src)
    row["cues_in"] = len(cues)
    if not cues:
        row.update(status="error", error="no cues")
        return row
    if clean:
        kept, dropped = clean_cues(cues, load_sidecar(src), min_confidence=min_confidence,
                                   cjk_source=source in ("zh", "ch", "ja"))
    else:
        kept, dropped = cues, []
    for c in dropped:
        log.info("%s: drop cue %d %s %r — %s", src.name, c.index, c.start, c.text.replace("\n", "⏎"), "; ".join(c.notes))
    for c in kept:
        if c.notes:
            log.info("%s: cue %d %s — %s", src.name, c.index, c.start, "; ".join(c.notes))
    row.update(dropped=len(dropped), cues_out=len(kept))
    if dry_run or client is None:
        log.info("%s: dry run — %d cues would be sent in %d request(s)", src.name, len(kept),
                 len(_chunks(kept, chunk)))
        row["status"] = "dry-run"
        return row
    before = dict(client.usage)
    translations, failed = translate_cues(kept, client, language=language, source=source, glossary=glossary,
                                          style=style, chunk=chunk, log_prefix=src.name + ": ")
    row["prompt_tokens"] = client.usage["prompt_tokens"] - before["prompt_tokens"]
    row["completion_tokens"] = client.usage["completion_tokens"] - before["completion_tokens"]
    write_srt(kept, translations, dst, bilingual=False)
    if bilingual:
        write_srt(kept, translations, dst.with_name(dst.name.replace(f".{lang_suffix(language)}.srt",
                                                                    f".{source}-{lang_suffix(language)}.srt")), bilingual=True)
    row.update(translated=len(translations), kept_source=len(failed), seconds=round(time.time() - t0, 1))
    if failed:
        row["status"] = "partial"
        row["error"] = f"untranslated (source kept): {failed[:20]}"
    log.info("%s: %d/%d cues translated -> %s (%.0fs, %d+%d tokens)", src.name, len(translations), len(kept),
             dst.name, row["seconds"], row["prompt_tokens"], row["completion_tokens"])
    return row


def find_srts(root: Path, suffix: str) -> list[Path]:
    if root.is_file():
        return [root]
    skip = re.compile(r"\.(vi|en|zh|ja|ko|th|fr|es|de|id|zh-vi|ch-vi)\.srt$", re.IGNORECASE)
    return sorted(p for p in root.glob("*.srt") if not skip.search(p.name) and not p.name.endswith(f".{suffix}.srt"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Translate OCR'ed .srt files with the DeepSeek API",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", help="an .srt file or a folder of them")
    p.add_argument("-o", "--output", help="output .srt (single file only; default <input>.<lang>.srt)")
    p.add_argument("--target", default="Vietnamese", help="target language name")
    p.add_argument("--source", default="zh", help="source language code: zh (Chinese), en, ja, ko")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--api-key", default=None, help="default: env DEEPSEEK_API_KEY")
    p.add_argument("--base-url", default=DEEPSEEK_BASE_URL)
    p.add_argument("--glossary", metavar="JSON", help='{"曼云": "Mạn Vân", ...}; used if it exists')
    p.add_argument("--build-glossary", action="store_true",
                   help="ask DeepSeek for the names/terms of the whole folder first and save them to --glossary "
                        "(default glossary.json in the folder); edit the file, then run again to translate")
    p.add_argument("--style", default="", help="extra style notes for the translator (tone, register, audience)")
    p.add_argument("--chunk", type=int, default=60, help="cues per request")
    p.add_argument("--workers", type=int, default=3, help="files translated in parallel")
    p.add_argument("--no-clean", action="store_true", help="keep low-confidence / no-CJK junk cues")
    p.add_argument("--min-confidence", type=float, default=0.9,
                   help="drop cues below this OCR confidence (needs the .ocr.json sidecar)")
    p.add_argument("--bilingual", action="store_true", help="also write <name>.zh-vi.srt with source + translation")
    p.add_argument("--force", action="store_true", help="redo files whose translation exists")
    p.add_argument("--dry-run", action="store_true", help="clean and chunk only, no API calls")
    p.add_argument("--report", help="CSV path (default translate_report.csv next to the files)")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    root = Path(a.input)
    suffix = lang_suffix(a.target)
    files = find_srts(root, suffix)
    if not files:
        sys.exit(f"no .srt files under {root}")
    folder = root.parent if root.is_file() else root

    api_key = a.api_key or os.environ.get("DEEPSEEK_API_KEY")
    client = None
    if not a.dry_run:
        if not api_key:
            sys.exit("set DEEPSEEK_API_KEY (or --api-key), or use --dry-run")
        client = DeepSeek(api_key, model=a.model, temperature=a.temperature, base_url=a.base_url)

    glossary_path = Path(a.glossary) if a.glossary else folder / "glossary.json"
    glossary: dict[str, str] = {}
    if a.build_glossary:
        if client is None:
            sys.exit("--build-glossary needs the API (drop --dry-run)")
        glossary = build_glossary(files, client, source=a.source)
        glossary_path.write_text(json.dumps(glossary, ensure_ascii=False, indent=1), encoding="utf-8")
        log.info("glossary: %d terms -> %s (edit it, then run again without --build-glossary)", len(glossary),
                 glossary_path)
        return 0
    if glossary_path.exists():
        glossary = json.loads(glossary_path.read_text(encoding="utf-8"))
        log.info("glossary: %d terms from %s", len(glossary), glossary_path)

    todo, skipped = [], 0
    for f in files:
        dst = Path(a.output) if (a.output and root.is_file()) else f.with_suffix(f".{suffix}.srt")
        if dst.exists() and not a.force:
            skipped += 1
        else:
            todo.append((f, dst))
    log.info("%d file(s), %d to do, %d already translated", len(files), len(todo), skipped)

    kwargs = dict(language=a.target, source=a.source, glossary=glossary, style=a.style, chunk=a.chunk,
                  clean=not a.no_clean, min_confidence=a.min_confidence, bilingual=a.bilingual, dry_run=a.dry_run)
    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = {ex.submit(translate_file, f, dst, client, **kwargs): f for f, dst in todo}
        for fut in as_completed(futs):
            f = futs[fut]
            try:
                rows.append(fut.result())
            except Exception as e:  # noqa: BLE001 — one bad file must not stop the batch
                log.error("FAILED %s: %s", f.name, e)
                rows.append(dict(file=f.name, output="", status="error", error=f"{type(e).__name__}: {e}"[:300]))
    rows.sort(key=lambda r: r["file"])

    if not a.dry_run and rows:
        report = Path(a.report) if a.report else folder / "translate_report.csv"
        exists = report.exists()
        with open(report, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=REPORT_FIELDS, extrasaction="ignore")
            if not exists:
                w.writeheader()
            w.writerows(rows)
        log.info("report: %s", report)
    ok = sum(1 for r in rows if r["status"] in ("ok", "dry-run"))
    partial = sum(1 for r in rows if r["status"] == "partial")
    usage = client.usage if client else {}
    log.info("done: %d ok, %d partial, %d failed, %d skipped, %.1f min, tokens %s", ok, partial,
             len(rows) - ok - partial, skipped, (time.time() - t0) / 60, usage)
    return 0 if ok + partial == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
