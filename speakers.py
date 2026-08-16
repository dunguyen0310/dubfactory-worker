"""Who says each line — speaker detection for subtitle files.

A 45-minute drama has a cast; rendering every cue in one voice is the ceiling on
how good the dub can sound. This module finds the speaker labels subtitlers
already write into the text, so each character can be cast to its own voice.

Three conventions are recognised, all at the start of a cue:

    MINH: Anh có chắc không?          name + colon
    [MINH] Anh có chắc không?         bracketed
    (MINH) Anh có chắc không?         parenthesised

plus italics (``<i>...</i>``), which subtitlers conventionally use for narration,
phone audio and inner monologue rather than for a named character.

**The label must be stripped, not spoken.** Everything here returns the clean
line alongside the speaker, and the clean line is what reaches the TTS engine.

The hard part is false positives: Vietnamese prose is full of colons that are
not speakers — "Chú ý:", "Lưu ý:", "Ví dụ:". Guarding on shape alone does not
separate those from "Minh:". So detection is **two-pass**: cue-level parsing
proposes candidates, then :func:`scan` keeps only labels that behave like
characters — recurring across the file, or written in the ALL-CAPS style that is
already an unambiguous subtitle convention for a speaker. A stray "Chú ý:" that
appears once is left alone and spoken as written.
"""

import re
import unicodedata
from collections import Counter

# Label styles, tried in order. Each must match at the very start of the cue and
# leave non-empty spoken text behind.
NAME_COLON_RE = re.compile(r"^\s*([^\s:][^:\n]{0,24}):\s+(?=\S)")
BRACKET_RE = re.compile(r"^\s*[\[\(]\s*([^\]\)\n]{1,25}?)\s*[\]\)]\s*(?=\S)")

ITALIC_RE = re.compile(r"<i>|</i>", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>|\{[^}]+\}")

# A label containing sentence punctuation is prose, not a name.
PROSE_PUNCT_RE = re.compile(r"[.!?;,]")

# Words that introduce prose rather than dialogue. Anything ending in one of
# these is rejected outright, whatever else it looks like.
PROSE_LEADS = {
    # Vietnamese
    "chú ý", "lưu ý", "ghi chú", "ví dụ", "chú thích", "cảnh báo", "gợi ý",
    "lưu ý rằng", "kết luận", "tóm lại", "nguồn", "dịch",
    # English
    "note", "warning", "caution", "example", "tip", "source", "translation",
    "subtitle", "subtitles", "caption", "captions", "narrator's note",
}

# The label used for italicised lines that carry no explicit name. Parenthesised
# so it can never collide with a real character name from the script.
VOICEOVER = "(voiceover)"

# A label recurring at least this many times reads as a character rather than a
# one-off prose lead-in.
MIN_RECURRENCE = 2

# Labels this long are almost certainly a sentence fragment, not a name.
MAX_LABEL_CHARS = 25


def _fold(s: str) -> str:
    """Casefold and strip diacritics, for comparing against PROSE_LEADS."""
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(
        c for c in unicodedata.normalize("NFD", s.casefold())
        if unicodedata.category(c) != "Mn"
    ).strip()


def is_shouted(label: str) -> bool:
    """ALL-CAPS labels are an unambiguous subtitle convention for a speaker.

    Checked on the letters only, so 'MINH' and 'BÀ LAN' qualify while 'Minh'
    does not. Requires at least one cased letter, so '???' is not 'shouted'.
    """
    letters = [c for c in label if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _plausible(label: str) -> bool:
    """Could this label be a character name at all?"""
    label = label.strip()
    if not label or len(label) > MAX_LABEL_CHARS:
        return False
    if PROSE_PUNCT_RE.search(label):
        return False
    if not any(c.isalpha() for c in label):
        return False
    if _fold(label) in PROSE_LEADS:
        return False
    # An unbalanced bracket means this is a fragment of a caption whose colon
    # happened to land mid-way — "(Kim Hyun Soo: đã bị bắt" would cast
    # "(Kim Hyun Soo" as a character. Balanced pairs stay legal ("MINH (VO)").
    for open_ch, close_ch in ("()", "[]", "{}"):
        if label.count(open_ch) != label.count(close_ch):
            return False
    # A label of many words is a sentence, not a name. Vietnamese names run to
    # four syllables ("Nguyễn Thị Bích Ngọc"), so allow up to five words.
    if len(label.split()) > 5:
        return False
    return True


def candidate(text: str):
    """Propose a speaker for one cue.

    Returns ``(label, clean_text)``. ``label`` is None when the cue carries no
    recognisable speaker prefix, in which case ``clean_text`` is the input with
    only markup removed. This is deliberately permissive — :func:`scan` decides
    which candidates are real.
    """
    raw = text or ""
    italic = bool(ITALIC_RE.search(raw))
    stripped = TAG_RE.sub("", raw).strip()

    for pattern in (BRACKET_RE, NAME_COLON_RE):
        m = pattern.match(stripped)
        if not m:
            continue
        label = m.group(1).strip()
        if not _plausible(label):
            continue
        spoken = stripped[m.end():].strip()
        if spoken:
            return label, spoken

    if italic and stripped:
        return VOICEOVER, stripped
    return None, stripped


def scan(texts):
    """Decide which candidate labels are real speakers, across a whole file.

    Takes an iterable of cue texts and returns ``(assignments, cast)`` where
    ``assignments`` is a list of ``(speaker, clean_text)`` parallel to the input
    and ``cast`` is a ``{speaker: line count}`` Counter, most frequent first.

    A candidate survives if it recurs (``MIN_RECURRENCE``) or is ALL-CAPS.
    Rejected candidates keep their label in the spoken text, because a line like
    "Chú ý: hôm nay giảm giá" is a sentence and dropping its first two words
    would silently corrupt the script.
    """
    texts = list(texts)
    proposals = [candidate(t) for t in texts]
    counts = Counter(label for label, _ in proposals if label)

    keep = {
        label for label, n in counts.items()
        if n >= MIN_RECURRENCE or is_shouted(label) or label == VOICEOVER
    }

    assignments, cast = [], Counter()
    for (label, clean), original in zip(proposals, texts):
        if label and label in keep:
            assignments.append((label, clean))
            cast[label] += 1
        else:
            # Not a speaker after all — keep the line exactly as written,
            # minus markup only.
            assignments.append((None, TAG_RE.sub("", original or "").strip()))
    return assignments, Counter(dict(cast.most_common()))


def strip_label(text: str, speaker: str | None) -> str:
    """Remove `speaker`'s label from `text`, returning what should be spoken.

    This is the counterpart to storing the speaker separately instead of
    destructively editing the script at upload time. `cues.source_text` keeps
    the line exactly as the subtitler wrote it and the label is removed here, at
    render time, from the stored speaker. Two things follow:

      * Detection can never delete words. If a label turns out to be prose
        rather than a character, clearing `cues.speaker` restores the full line
        with nothing lost — there is no original to recover because it was
        never overwritten.
      * A mismatch is a no-op. If `text` does not actually start with
        `speaker`'s label, the line is returned untouched rather than having
        its first few words guessed away.
    """
    clean = TAG_RE.sub("", text or "").strip()
    if not speaker or speaker == VOICEOVER:
        return clean
    for pattern in (BRACKET_RE, NAME_COLON_RE):
        m = pattern.match(clean)
        if m and m.group(1).strip() == speaker:
            rest = clean[m.end():].strip()
            if rest:
                return rest
    return clean


def describe(cast) -> str:
    """One-line summary for logs and CLI output."""
    if not cast:
        return "no speaker labels found — single voice"
    parts = [f"{name} ({n})" for name, n in cast.most_common()]
    return f"{len(cast)} speaker(s): " + ", ".join(parts)


if __name__ == "__main__":
    import argparse
    import srt_dub as S

    ap = argparse.ArgumentParser(
        description="Show the cast a subtitle file implies.")
    ap.add_argument("srt")
    ap.add_argument("--show-lines", action="store_true",
                    help="print every line with the speaker it was assigned")
    args = ap.parse_args()

    cues = S.parse_srt(args.srt)
    # .raw keeps the markup parse_srt strips from .text; italics are a speaker
    # signal, so detection must run on the unstripped line.
    assignments, cast = scan([c.raw or c.text for c in cues])
    print(f"parsed {len(cues)} cues")
    print(describe(cast))
    if cast:
        width = max(len(n) for n in cast)
        print()
        for name, n in cast.most_common():
            share = 100 * n / max(len(cues), 1)
            print(f"  {name:<{width}}  {n:>4} lines  {share:5.1f}%")
    if args.show_lines:
        print()
        for cue, (who, clean) in zip(cues, assignments):
            print(f"  {cue.index:>4}  {who or '-':<12}  {clean[:60]}")
