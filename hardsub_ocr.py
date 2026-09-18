#!/usr/bin/env python
"""
hardsub_ocr.py — burned-in (hardcoded) subtitles -> .srt, headless.

    python hardsub_ocr.py video.mp4                       # -> video.srt next to the video
    python hardsub_ocr.py video.mp4 -o out.srt --lang ch  # Chinese + English subtitles
    python hardsub_ocr.py video.mp4 --lang vi             # Vietnamese (Latin-script model)

A Python port of Subtitle Edit 5.2's "Video -> OCR burned-in subtitle"
pipeline (src/ui/Features/Video/VideoOcr in the SubtitleEdit repo), which is
GUI-only there.  Same stages and the same tuned defaults, so a video that OCRs
well in Subtitle Edit OCRs the same here — but this runs from a script over a
folder of hundreds of videos.

    1. ffmpeg samples the scan area (default: bottom third, full width) at
       --fps frames per second, downscaled to --max-width, as JPEGs.  Each
       sampled frame's real timestamp is taken from ffmpeg's showinfo.
    2. Frames are collapsed into runs of near-identical frames ("groups") by
       comparing a 96-px-wide *brightness mask* (pixels >= --brightness-min),
       Jaccard similarity >= --image-sim.  Runs with no bright pixels are blank.
    3. One representative frame per group (the middle one) is OCR'ed with
       RapidOCR (PaddleOCR models on ONNX Runtime).  The frame is masked first
       — everything below the brightness minimum is blacked out — so the
       detector only sees the bright subtitle text, not scene text.
    4. Text that sits at the same place in many frames (a channel logo or
       watermark inside the scan area) is removed automatically; junk boxes
       (no letters, tiny fragments, lone low-confidence characters) are dropped.
    5. Consecutive groups whose text is near-identical (Levenshtein
       >= --text-sim, gap <= --max-gap) merge into one subtitle line; the text
       variant on screen longest wins the vote; blips < --min-duration drop;
       a line an OCR miss cut in two is re-joined (--bridge-gap).
    6. Timing refinement: the coarse scan snaps every boundary to the fps grid
       (200 ms at 5 fps).  For each start and end, the one coarse interval
       around it is re-decoded at the video's native frame rate and the exact
       frame where the text appears/disappears is found by mask similarity.
       No OCR involved — two tiny ffmpeg calls per subtitle.
    7. .srt is written, plus a .ocr.json sidecar with per-cue confidence.

Requirements: ffmpeg + ffprobe on PATH; pip install rapidocr onnxruntime
opencv-python-headless numpy rapidfuzz.  Models download on first use.
CPU is fine: only one frame per subtitle is OCR'ed (~0.2 s each).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz.distance import Levenshtein

# ---------------------------------------------------------------- defaults
# Grouping / merging defaults are Subtitle Edit's (src/ui/Logic/Config/SeVideoOcr.cs).
# max_width is 1280 rather than SE's 720: measured on a 720p test clip, the
# recogniser reads full-resolution glyphs better, and only one frame per
# subtitle is OCR'ed, so the extra ~50 ms per frame is cheap.
DEFAULTS = dict(
    fps=5,
    image_sim=92,
    text_sim=80,
    max_gap=250,
    min_duration=250,
    bridge_gap=1000,      # re-join same-text lines separated by an OCR miss up to this gap
    brightness_min=190,
    max_width=1280,
    crop=(0.0, 200.0 / 3.0, 100.0, 100.0 / 3.0),   # x%, y%, w%, h%: bottom third (fallback)
    auto_crop=True,       # find the subtitle band first and scan only that
    search=(0.0, 40.0, 100.0, 60.0),               # where auto-crop looks for subtitles: bottom 60%
    det="v6-small",
    rec="auto",
    det_limit=960,        # detector resizes so the longer side is <= this (never upscales a strip)
    unclip=2.2,           # detector box expansion: whole lines instead of word fragments
    min_score=0.5,
    static_share=0.4,     # text at one position in >= this share of text frames is a watermark
)
THUMB_W = 96           # grouping thumbnail width
MASK_SRC_W = 360       # mask is thresholded at this width, then max-pooled to THUMB_W
BLANK_FRACTION = 0.002
REFINE_PAD_MS = 60
REFINE_FALLBACK_STEP_MS = 20
REFINE_MIN_SIM = 30    # below this a window frame is not the subtitle whatever the other reference says
REFINE_MIN_JUMP = 10   # similarity points: smallest change that counts as the text appearing/leaving
STATIC_MIN_GROUPS = 8
STATIC_MIN_OCCURRENCES = 5
STATIC_POS_TOL = 0.04  # fraction of frame width/height

# RapidOCR recognition model per --lang.  "ch" reads Chinese *and* English
# (digits, punctuation); "latin" covers Vietnamese, French, Spanish, ...;
# "en" is English-only and a little more accurate on pure English.
LANG_ALIASES = {
    "vi": "latin", "vietnamese": "latin", "fr": "latin", "es": "latin", "de": "latin",
    "pt": "latin", "it": "latin", "id": "latin", "nl": "latin", "pl": "latin", "tr": "latin",
    "zh": "ch", "chinese": "ch", "cn": "ch", "zh-cn": "ch",
    "zh-tw": "chinese_cht", "cht": "chinese_cht", "traditional": "chinese_cht",
    "ja": "japan", "japanese": "japan", "jp": "japan",
    "ko": "korean", "kr": "korean",
    "ru": "cyrillic", "uk": "cyrillic", "bg": "cyrillic",
    "ar": "arabic", "hi": "devanagari", "gr": "el", "thai": "th",
}
# (version, model_type) candidates per recognition language, tried in order —
# the models that exist in rapidocr 3.9's catalogue.
REC_CANDIDATES = {
    "ch": [("v6", "small"), ("v5", "mobile"), ("v4", "mobile")],
    "en": [("v5", "mobile"), ("v4", "mobile")],
    "japan": [("v4", "mobile"), ("v5", "mobile")],
    "chinese_cht": [("v4", "mobile"), ("v5", "mobile")],
}
REC_FALLBACK = [("v5", "mobile"), ("v4", "mobile")]
NO_SPACE_SCRIPTS = ("ch", "chinese_cht", "japan", "th")

log = logging.getLogger("hardsub")
_SHOWINFO = re.compile(r" n:\s*(\d+).*?\spts_time:([0-9.]+)")


def imread(path) -> np.ndarray | None:
    """cv2.imread cannot open non-ASCII paths on Windows (a video called
    凤凰特工001 made every frame unreadable); decode from bytes instead."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------- data
@dataclass
class FrameGroup:
    start: int                 # first sampled-frame index
    end: int                   # last sampled-frame index (inclusive)
    blank: bool
    rep_file: str = ""         # representative frame (middle of the run)
    t_start: float = 0.0       # ms: timestamp of the first sampled frame
    t_end: float = 0.0         # ms: timestamp of the sampled frame after the last one
    boxes: list = field(default_factory=list)
    text: str = ""
    confidence: float = 1.0


@dataclass
class Line:
    start_ms: float
    end_ms: float
    text: str
    start_index: int = 0       # first / last coarse frame of the line (for refinement)
    end_index: int = 0
    confidence: float = 1.0
    votes: dict = field(default_factory=dict)
    conf_weight: float = 0.0
    conf_sum: float = 0.0

    def majority(self) -> str:
        return max(self.votes.items(), key=lambda kv: kv[1])[0]

    def add(self, text: str, weight: float, conf: float) -> None:
        self.votes[text] = self.votes.get(text, 0.0) + weight
        self.conf_sum += conf * weight
        self.conf_weight += weight

    def absorb(self, other: "Line") -> None:
        self.end_ms = other.end_ms
        self.end_index = other.end_index
        for t, w in other.votes.items():
            self.votes[t] = self.votes.get(t, 0.0) + w
        self.conf_sum += other.conf_sum
        self.conf_weight += other.conf_weight
        self.text = self.majority()
        self.confidence = self.conf_sum / self.conf_weight if self.conf_weight else self.confidence


# ---------------------------------------------------------------- media probe
_FPS_MODE: list[str] | None = None


def fps_mode_args() -> list[str]:
    """`-fps_mode passthrough`, or its pre-5.1 spelling `-vsync passthrough`.

    One output frame per source frame is load-bearing here (see _decode_window),
    and the flag was renamed in ffmpeg 5.1. The Dub Factory worker runs on
    Colab's Ubuntu image, which ships ffmpeg 4.4, so the old spelling has to
    work too. Probed once per process from `ffmpeg -version`; an unparseable
    banner is assumed to be a current build.
    """
    global _FPS_MODE
    if _FPS_MODE is None:
        _FPS_MODE = ["-fps_mode", "passthrough"]
        try:
            banner = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout
            m = re.search(r"ffmpeg version (?:n)?(\d+)\.(\d+)", banner or "")
            if m and (int(m.group(1)), int(m.group(2))) < (5, 1):
                _FPS_MODE = ["-vsync", "passthrough"]
                log.info("ffmpeg %s.%s: using -vsync passthrough (-fps_mode needs >= 5.1)",
                         m.group(1), m.group(2))
        except OSError:
            pass                                    # no ffmpeg at all: the first real call reports it
    return list(_FPS_MODE)


def probe(video: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate",
         "-show_entries", "format=duration", "-of", "json", video],
        capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    native_fps = 0.0
    for key in ("avg_frame_rate", "r_frame_rate"):
        num, den = (s.get(key) or "0/1").split("/")
        if float(den or 0) and float(num):
            native_fps = float(num) / float(den)
            break
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": native_fps or 25.0,
            "duration": float(j["format"].get("duration") or 0.0)}


def crop_filter(info: dict, crop_pct, max_width: int) -> tuple[str, tuple[int, int, int, int]]:
    """crop=W:H:X:Y[,scale=max_width:-2] from percentages of the frame."""
    x_pct, y_pct, w_pct, h_pct = crop_pct
    W, H = info["width"], info["height"]
    x = int(round(W * x_pct / 100.0)) // 2 * 2
    y = int(round(H * y_pct / 100.0)) // 2 * 2
    w = max(2, min(int(round(W * w_pct / 100.0)) // 2 * 2, W - x))
    h = max(2, min(int(round(H * h_pct / 100.0)) // 2 * 2, H - y))
    vf = f"crop={w}:{h}:{x}:{y}"
    if max_width and w > max_width:
        vf += f",scale={max_width}:-2"
    return vf, (x, y, w, h)


def _parse_showinfo(stderr: bytes) -> dict[int, float]:
    """{frame n: pts ms} from ffmpeg's showinfo filter output."""
    times = {}
    for ln in stderr.decode("utf-8", "replace").splitlines():
        if "pts_time:" in ln:
            m = _SHOWINFO.search(ln)
            if m:
                times[int(m.group(1))] = float(m.group(2)) * 1000.0
    return times


# ---------------------------------------------------------------- 1. extract
def extract_frames(video: str, frames_dir: Path, fps: int, crop_vf: str) -> tuple[list[str], list[float]]:
    """Sampled frames as JPEG files plus each frame's timestamp in ms (from
    showinfo, so a stream whose first frame is not at 0 is labelled right)."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(frames_dir / "img%06d.jpg")
    # `select` keeps the first source frame at least 1/fps after the previously kept
    # one, so every sampled frame carries its *own* timestamp (showinfo).  ffmpeg's
    # fps filter instead emits, for each output slot, the last source frame that
    # rounds into it — measured on a 24 fps clip, every sample was 40-90 ms later
    # than its label, which put every coarse start and end late by that much.
    interval = 1.0 / fps - 0.001
    select = f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{interval:.4f})'"
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-y",
           "-i", video, "-vf", f"{select},{crop_vf},showinfo", *fps_mode_args(),
           "-q:v", "2", pattern]
    log.info("extracting frames: -vf \"%s,%s\"", select, crop_vf)
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + p.stderr.decode("utf-8", "replace")[-1500:])
    files = sorted(str(f) for f in frames_dir.glob("img*.jpg"))
    if not files:
        raise RuntimeError("ffmpeg produced no frames — check the crop area / video")
    times = _parse_showinfo(p.stderr)
    step = 1000.0 / fps
    frame_times = [times.get(i, i * step) for i in range(len(files))]
    return files, frame_times


# ---------------------------------------------------------------- 2. group
def _luma(img_bgr: np.ndarray) -> np.ndarray:
    b = img_bgr[:, :, 0].astype(np.int32)
    g = img_bgr[:, :, 1].astype(np.int32)
    r = img_bgr[:, :, 2].astype(np.int32)
    return (r * 299 + g * 587 + b * 114) // 1000


def _pool_starts(src: int, dst: int) -> np.ndarray:
    """First source index of each destination cell for the mapping
    cell = min(dst-1, src_index * dst // src)  (Subtitle Edit's MakePooledMask)."""
    cells = np.minimum(dst - 1, np.arange(src) * dst // src)
    starts = np.searchsorted(cells, np.arange(dst), side="left")
    return np.minimum(starts, src - 1)


def make_mask(img_bgr: np.ndarray | None, brightness_min: int) -> np.ndarray | None:
    """96-px-wide pooled brightness mask (uint8 0/255), or a grayscale thumbnail
    when brightness_min == 0.  Mirrors VideoOcrFrameGrouper.MakeThumbnail."""
    if img_bgr is None or img_bgr.size == 0:
        return None
    h, w = img_bgr.shape[:2]
    if brightness_min > 0:
        if w > MASK_SRC_W:
            nh = max(1, int(round(h * MASK_SRC_W / w)))
            img_bgr = cv2.resize(img_bgr, (MASK_SRC_W, nh), interpolation=cv2.INTER_AREA)
            h, w = img_bgr.shape[:2]
        bright = (_luma(img_bgr) >= brightness_min).astype(np.uint8)
        th = max(1, int(round(h * THUMB_W / w)))
        pooled = np.maximum.reduceat(bright, _pool_starts(w, THUMB_W), axis=1)
        pooled = np.maximum.reduceat(pooled, _pool_starts(h, th), axis=0)
        return (pooled * 255).astype(np.uint8)
    th = max(1, int(round(h * THUMB_W / w)))
    small = cv2.resize(img_bgr, (THUMB_W, th), interpolation=cv2.INTER_LINEAR)
    return _luma(small).astype(np.uint8)


def is_blank(mask: np.ndarray) -> bool:
    return int(np.count_nonzero(mask)) < mask.size * BLANK_FRACTION


def mask_similarity(a: np.ndarray, b: np.ndarray) -> int:
    """Jaccard overlap of two bright-pixel masks, percent."""
    if a.shape != b.shape or a.size == 0:
        return 0
    ia, ib = a > 0, b > 0
    union = int(np.count_nonzero(ia | ib))
    if union == 0:
        return 100
    return int(round(np.count_nonzero(ia & ib) * 100.0 / union))


def gray_similarity(a: np.ndarray, b: np.ndarray) -> int:
    if a.shape != b.shape or a.size == 0:
        return 0
    mean_diff = np.abs(a.astype(np.int32) - b.astype(np.int32)).mean()
    return int(round(100.0 - mean_diff * 100.0 / 255.0))


def similar(ref: np.ndarray, m: np.ndarray, brightness_min: int, image_sim: int) -> bool:
    if brightness_min > 0:
        return not is_blank(m) and mask_similarity(ref, m) >= image_sim
    return gray_similarity(ref, m) >= image_sim


def group_frames(files: list[str], frame_times: list[float], fps: int, brightness_min: int,
                 image_sim: int) -> list[FrameGroup]:
    groups: list[FrameGroup] = []
    current: FrameGroup | None = None
    current_files: list[str] = []
    last = None

    def close():
        if current is not None and current_files:
            current.rep_file = current_files[len(current_files) // 2]
            groups.append(current)

    for index, f in enumerate(files):
        thumb = make_mask(imread(f), brightness_min)
        if thumb is None:
            if current is not None:          # unreadable frame stays in the run
                current.end = index
                current_files.append(f)
            continue
        blank = brightness_min > 0 and is_blank(thumb)
        same = (current is not None and last is not None and last.shape == thumb.shape
                and current.blank == blank
                and (current.blank or
                     (mask_similarity(last, thumb) if brightness_min > 0
                      else gray_similarity(last, thumb)) >= image_sim))
        if same:
            current.end = index
            current_files.append(f)
        else:
            close()
            current = FrameGroup(start=index, end=index, blank=blank)
            current_files = [f]
        last = thumb
    close()

    step = 1000.0 / fps
    for g in groups:
        g.t_start = frame_times[g.start]
        g.t_end = frame_times[g.end + 1] if g.end + 1 < len(frame_times) else frame_times[g.end] + step
    return groups


# ---------------------------------------------------------------- 3. OCR
def masked_copy(img_bgr: np.ndarray, brightness_min: int) -> np.ndarray:
    """Black out everything below the brightness minimum (dilated by 2 px so
    anti-aliased glyph edges survive) — VideoOcrFrameGrouper.WriteMaskedCopy."""
    keep = (_luma(img_bgr) >= brightness_min).astype(np.uint8)
    keep = cv2.dilate(keep, np.ones((5, 5), np.uint8))
    out = img_bgr.copy()
    out[keep == 0] = 0
    return out


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x2E80 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF or 0xF900 <= o <= 0xFAFF
            or 0xFF00 <= o <= 0xFFEF or 0x3040 <= o <= 0x30FF)


def _content_len(text: str) -> int:
    return sum(1 for ch in text if ch.isalnum() or _is_cjk(ch))


# Characters a bright horizontal edge in the scene reads as: the one- and two-
# stroke numerals, dashes, underscores.  A box made only of these is a line in
# the picture, not text.
STROKE_CHARS = set("一二丨ー—–-_=~·•.,:;'\"`´ 　")


def _stroke_only(text: str) -> bool:
    return bool(text) and all(ch in STROKE_CHARS for ch in text)


def _preload_cuda_dlls() -> None:
    """Load the CUDA / cuDNN libraries shipped as nvidia-* pip packages (Colab, most
    CUDA venvs) before the first session, so ONNX Runtime's CUDA provider can find
    them without LD_LIBRARY_PATH.  No-op on CPU-only builds or old onnxruntime."""
    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(cuda=True, cudnn=True)
    except Exception as e:  # noqa: BLE001 — best effort; rapidocr will report the provider actually used
        log.warning("could not preload CUDA libraries: %s", e)


class Ocr:
    """RapidOCR wrapper: picks models per language, returns text boxes."""

    def __init__(self, lang: str = "ch", det: str = DEFAULTS["det"], rec: str = DEFAULTS["rec"],
                 det_limit: int = DEFAULTS["det_limit"], unclip: float = DEFAULTS["unclip"],
                 gpu: bool = False, threads: int | None = None):
        from rapidocr import RapidOCR, LangRec, OCRVersion, ModelType
        low = lang.strip().lower()
        self.lang = LANG_ALIASES.get(low, low)
        try:
            rec_lang = LangRec(self.lang)
        except ValueError:
            raise SystemExit(f"unknown --lang {lang!r}; use one of: "
                             + ", ".join(m.value for m in LangRec) + " or an alias like vi/zh/ko/ja")
        self.unclip = unclip
        versions = {"v4": OCRVersion.PPOCRV4, "v5": OCRVersion.PPOCRV5, "v6": OCRVersion.PPOCRV6}
        mtypes = {"mobile": ModelType.MOBILE, "server": ModelType.SERVER, "small": ModelType.SMALL,
                  "medium": ModelType.MEDIUM, "tiny": ModelType.TINY}
        dv, dm = det.split("-")
        if rec == "auto":
            candidates = REC_CANDIDATES.get(self.lang, REC_FALLBACK)
        elif "-" in rec:
            candidates = [tuple(rec.split("-"))]
        else:                       # just a model type, e.g. "server": try the newest versions
            candidates = [(v, rec) for v in ("v5", "v4", "v6")]
        errors = []
        for rv, rm in candidates:
            params = {
                # RapidOCR logs a WARNING for every frame without text — the normal case
                # here (most sampled frames have no subtitle) — so keep only errors.
                "Global.log_level": "error",
                "Det.ocr_version": versions[dv], "Det.model_type": mtypes[dm],
                "Det.limit_type": "max", "Det.limit_side_len": int(det_limit),
                "Rec.lang_type": rec_lang, "Rec.ocr_version": versions[rv], "Rec.model_type": mtypes[rm],
            }
            # rapidocr >= 3.x copies EngineConfig.<engine> into Det/Cls/Rec.engine_cfg *after*
            # applying params, so per-module keys ("Det.engine_cfg.onnxruntime.use_cuda") are
            # silently overwritten (verified on rapidocr 3.9.2); the global keys stick.
            if gpu:   # needs `pip install onnxruntime-gpu` (CUDA 12 + cuDNN 9); falls back to CPU
                params["EngineConfig.onnxruntime.use_cuda"] = True
                _preload_cuda_dlls()
            if threads:   # ONNX Runtime otherwise takes every core: several processes at once
                params["EngineConfig.onnxruntime.intra_op_num_threads"] = int(threads)
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.engine = RapidOCR(params=params)
                self.models = f"det {dv}-{dm}, rec {self.lang} {rv}-{rm}"
                return
            except Exception as e:  # noqa: BLE001 — model combination not in the catalogue
                errors.append(f"{rv}-{rm}: {str(e).strip().splitlines()[-1][:120]}")
        raise SystemExit(f"no recognition model for language {self.lang!r}:\n  " + "\n  ".join(errors))

    def detect(self, img_bgr: np.ndarray) -> list[tuple[float, float, float, float]]:
        """Text boxes only (x0, y0, x1, y1), no recognition — for the band scan."""
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            res = self.engine(img_bgr, use_det=True, use_cls=False, use_rec=False, unclip_ratio=self.unclip)
        boxes = getattr(res, "boxes", None)
        if boxes is None or not len(boxes):
            return []
        out = []
        for box in boxes:
            pts = np.asarray(box, dtype=np.float32)
            out.append((float(pts[:, 0].min()), float(pts[:, 1].min()),
                        float(pts[:, 0].max()), float(pts[:, 1].max())))
        return out

    def __call__(self, img_bgr: np.ndarray, min_score: float) -> list[dict]:
        # RapidOCR keeps the last use_det/use_cls/use_rec it was given, so after a
        # detect() call the flags must be set back explicitly.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            res = self.engine(img_bgr, use_det=True, use_cls=True, use_rec=True, unclip_ratio=self.unclip)
        if res is None or getattr(res, "boxes", None) is None or not len(res.boxes):
            return []
        h, w = img_bgr.shape[:2]
        out = []
        for box, txt, score in zip(res.boxes, res.txts, res.scores):
            txt = (txt or "").strip()
            if not txt or score < min_score:
                continue
            pts = np.asarray(box, dtype=np.float32)
            x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
            y0, y1 = float(pts[:, 1].min()), float(pts[:, 1].max())
            out.append(dict(txt=txt, score=float(score), x0=x0, x1=x1, y0=y0, y1=y1,
                            h=max(1.0, y1 - y0), cy_px=(y0 + y1) / 2,
                            cx=(x0 + x1) / 2 / w, cy=(y0 + y1) / 2 / h))
        return out


def filter_boxes(boxes: list[dict], ignore: list[re.Pattern]) -> list[dict]:
    """Drop what is never subtitle text: boxes without a letter/digit, fragments
    much smaller than the frame's text, lone low-confidence characters, and
    anything matching --ignore-text."""
    boxes = [b for b in boxes if _content_len(b["txt"]) > 0 and not _stroke_only(b["txt"])]
    boxes = [b for b in boxes if not any(p.search(b["txt"]) for p in ignore)]
    if len(boxes) >= 2:
        med_h = statistics.median(b["h"] for b in boxes)
        boxes = [b for b in boxes if b["h"] >= 0.5 * med_h]
    return [b for b in boxes
            if not (len(b["txt"]) == 1 and not _is_cjk(b["txt"]) and b["score"] < 0.98)]


def drop_side_text(boxes: list[dict]) -> list[dict]:
    """A couple of characters far from the centre line of the scan area is a
    graphic — a vertical "未完待续" end card at the right edge, a channel mark —
    not a subtitle: subtitle lines, however short, are centred."""
    if boxes and sum(_content_len(b["txt"]) for b in boxes) <= 4             and all(abs(b["cx"] - 0.5) > 0.2 for b in boxes):
        return []
    return boxes


def _norm(text: str) -> str:
    return "".join(ch.lower() for ch in text if not ch.isspace())


def remove_static_text(groups: list[FrameGroup], share: float, total_frames: int) -> list[str]:
    """Remove logos / watermarks / channel names inside the scan area: the same
    text at the same position, seen again and again through the video.

    A sighting is a *run* of consecutive groups showing the text — a subtitle
    that stays on screen while a busy background keeps splitting the frame
    groups is still one sighting.  A candidate needs >= STATIC_MIN_OCCURRENCES
    separate sightings spread over the video at a stable position, and then
    either (a) a large share of all sightings of any text, (b) an off-centre
    position — subtitles are centred, logos sit in a corner — or (c) usually
    showing up next to other text, which a subtitle never does."""
    texted = [g for g in groups if g.boxes]
    if share <= 0 or len(texted) < STATIC_MIN_GROUPS:
        return []
    text_frames = sum(g.end - g.start + 1 for g in texted)
    info: dict[str, dict] = {}
    last_end: dict[str, int] = {}
    for g in texted:
        seen: dict[str, tuple[float, float]] = {}
        for b in g.boxes:
            seen.setdefault(_norm(b["txt"]), (b["cx"], b["cy"]))
        substantial = [k for k in seen if _content_len(k) >= 3]
        for key, (cx, cy) in seen.items():
            d = info.setdefault(key, dict(runs=[], frames=0, pos=[], companions=[]))
            if d["runs"] and g.start - last_end.get(key, -10) <= 5:   # ~1 s at 5 fps: same sighting
                d["runs"][-1][1] = g.end
            else:
                d["runs"].append([g.start, g.end])
            last_end[key] = g.end
            d["frames"] += g.end - g.start + 1
            d["pos"].append((cx, cy))
            d["companions"].extend((other, g.start) for other in substantial if other != key)
    span_frames = max(1, total_frames)
    static: dict[str, tuple[float, float]] = {}
    for key, d in info.items():
        mx = statistics.median(p[0] for p in d["pos"])
        my = statistics.median(p[1] for p in d["pos"])
        stable = sum(1 for p in d["pos"] if abs(p[0] - mx) <= STATIC_POS_TOL and abs(p[1] - my) <= STATIC_POS_TOL)
        if stable < 0.8 * len(d["pos"]):
            continue
        # Distinct texts this one was seen next to (OCR jitter of one companion
        # clustered away), and how much of the video those sightings span.  A
        # subtitle line shares the screen with at most its partner line; a logo
        # shares it with every subtitle of the film.
        reps: list[list] = []
        for text, frame in d["companions"]:
            for r in reps:
                if text_similarity(r[0], text) >= 80:
                    r[1], r[2] = min(r[1], frame), max(r[2], frame)
                    break
            else:
                reps.append([text, frame, frame])
        companion_span = (max(r[2] for r in reps) - min(r[1] for r in reps)) / span_frames if reps else 0.0
        many_companions = len(reps) >= 3 and companion_span >= 0.25
        n_runs = len(d["runs"])
        spread = (d["runs"][-1][0] - d["runs"][0][0]) / span_frames
        off_centre = abs(mx - 0.5) > 0.15
        persistent = d["frames"] >= share * text_frames and n_runs >= 2
        recurring = (n_runs >= STATIC_MIN_OCCURRENCES and spread >= 0.25
                     and (off_centre or persistent or len(reps) >= 2))
        if many_companions or persistent or recurring:
            static[key] = (mx, my)
    if not static:
        return []
    for g in texted:
        g.boxes = [b for b in g.boxes
                   if not (_norm(b["txt"]) in static
                           and abs(b["cx"] - static[_norm(b["txt"])][0]) <= STATIC_POS_TOL
                           and abs(b["cy"] - static[_norm(b["txt"])][1]) <= STATIC_POS_TOL)]
    return sorted(static)


def assemble_text(boxes: list[dict], lang: str) -> tuple[str, float]:
    """Order boxes into reading order: rows by vertical position (top first),
    left to right inside a row.  Returns (text, length-weighted confidence)."""
    if not boxes:
        return "", 0.0
    rows: list[list[dict]] = []
    for it in sorted(boxes, key=lambda d: d["cy_px"]):
        if rows:
            row = rows[-1]
            # same row when the box shares at least half of the smaller height with
            # the row's vertical span — a stray tall box cannot swallow the next line
            y0 = float(np.median([r["y0"] for r in row]))
            y1 = float(np.median([r["y1"] for r in row]))
            overlap = min(it["y1"], y1) - max(it["y0"], y0)
            if overlap >= 0.5 * min(it["h"], y1 - y0):
                row.append(it)
                continue
        rows.append([it])
    lines = []
    for row in rows:
        row.sort(key=lambda d: d["x0"])
        text = ""
        for it in row:
            if text:
                nospace = lang in NO_SPACE_SCRIPTS and _is_cjk(text[-1]) and _is_cjk(it["txt"][0])
                text += ("" if nospace else " ") + it["txt"]
            else:
                text = it["txt"]
        lines.append(text)
    total = sum(len(b["txt"]) for b in boxes) or 1
    conf = sum(b["score"] * len(b["txt"]) for b in boxes) / total
    return "\n".join(lines), conf


def ocr_groups(groups: list[FrameGroup], ocr: Ocr, brightness_min: int, use_mask: bool,
               min_score: float, ignore: list[re.Pattern], side_text: bool = False,
               progress=None) -> None:
    todo = [g for g in groups if not g.blank]
    for n, g in enumerate(todo, 1):
        img = imread(g.rep_file)
        if img is None:
            continue
        if use_mask and brightness_min > 0:
            img = masked_copy(img, brightness_min)
        g.boxes = filter_boxes(ocr(img, min_score), ignore)
        if not side_text:
            g.boxes = drop_side_text(g.boxes)
        if progress:
            progress(n, len(todo))


# ---------------------------------------------------------------- 4. lines
def text_similarity(a: str, b: str) -> int:
    s1, s2 = _norm(a), _norm(b)
    if not s1 and not s2:
        return 100
    if not s1 or not s2:
        return 0
    return int(round(100.0 * Levenshtein.normalized_similarity(s1, s2)))


def build_lines(groups: list[FrameGroup], text_sim: int, max_gap: int, min_duration: int) -> list[Line]:
    """Port of VideoOcrLineBuilder.Build: merge, vote, drop blips."""
    work: list[Line] = []
    current: Line | None = None
    previous: Line | None = None
    for g in sorted(groups, key=lambda g: g.start):
        text = (g.text or "").strip()
        if g.blank or not text:
            continue
        s, e = g.t_start, g.t_end
        weight = (e - s) * min(1.0, max(0.1, g.confidence))
        if current is not None and s - current.end_ms <= max_gap \
                and text_similarity(current.majority(), text) >= text_sim:
            current.end_ms, current.end_index = e, g.end
            current.add(text, weight, g.confidence)
        elif (current is not None and current.end_ms - current.start_ms < min_duration
              and previous is not None and s - previous.end_ms <= max_gap
              and text_similarity(previous.majority(), text) >= text_sim):
            # one junk observation must not sever the chain (Subtitle Edit's comment:
            # a jersey number read over "Wait." between two clean "Wait." reads)
            previous.end_ms, previous.end_index = e, g.end
            previous.add(text, weight, g.confidence)
            current, previous = previous, None
        else:
            previous = current
            current = Line(start_ms=s, end_ms=e, text=text, start_index=g.start, end_index=g.end)
            current.add(text, weight, g.confidence)
            work.append(current)
    out = []
    for ln in work:
        if ln.end_ms - ln.start_ms >= min_duration:
            ln.text = ln.majority()
            ln.confidence = ln.conf_sum / ln.conf_weight if ln.conf_weight else 0.0
            out.append(ln)
    return out


def filter_lines(lines: list[Line], min_duration: int) -> list[Line]:
    """Drop scene junk that survived as a line: a lone character the OCR was
    not sure about, or a 1-2 character flash."""
    out = []
    for ln in lines:
        n = _content_len(ln.text)
        dur = ln.end_ms - ln.start_ms
        if n == 0:
            continue
        if n == 1 and (ln.confidence < 0.95 or (not _is_cjk(ln.text.strip()[0]) and dur < 1000)):
            continue
        if n < 3 and dur < max(600, 2 * min_duration):
            continue
        out.append(ln)
    return out


def bridge_lines(lines: list[Line], text_sim: int, max_gap: int, bridge_gap: int) -> list[Line]:
    """Re-join a subtitle that an OCR miss (bright background for a moment)
    split in two: near-identical text with a short gap between the pieces.
    Also glue a sub-second fragment whose text half-matches a long neighbour
    it touches — the same subtitle read badly for a moment, scene junk mixed
    in — onto that neighbour; the long neighbour's text wins the vote."""
    out: list[Line] = []
    need = max(text_sim, 90)
    for ln in lines:
        if out:
            prev = out[-1]
            gap = ln.start_ms - prev.end_ms
            sim = text_similarity(prev.text, ln.text)
            if 0 <= gap <= bridge_gap and sim >= need:
                prev.absorb(ln)
                continue
            short, prev_short = ln.end_ms - ln.start_ms < 800, prev.end_ms - prev.start_ms < 800
            if 0 <= gap <= max_gap and sim >= 55 and short != prev_short:
                if short:
                    prev.absorb(ln)
                else:
                    ln.start_ms, ln.start_index = prev.start_ms, prev.start_index
                    for t, w in prev.votes.items():
                        ln.votes[t] = ln.votes.get(t, 0.0) + w
                    ln.conf_sum += prev.conf_sum
                    ln.conf_weight += prev.conf_weight
                    ln.text = ln.majority()
                    ln.confidence = ln.conf_sum / ln.conf_weight if ln.conf_weight else ln.confidence
                    out[-1] = ln
                continue
        out.append(ln)
    return out


def extend_lines(lines: list[Line], groups: list[FrameGroup], files: list[str],
                 frame_times: list[float], fps: int, brightness_min: int, image_sim: int,
                 text_sim: int, ocr: "Ocr", use_mask: bool, min_score: float,
                 ignore: list[re.Pattern], lang: str) -> int:
    """Grow each line over neighbouring sampled frames that carry the same
    text but were never read as such: the OCR ran on the *middle* frame of a
    group, so a cue's first or last sample can sit inside a group whose
    middle frame has no text (small text over a bright, moving background).
    Without this such a cue starts or ends a whole sample or more off.

    A neighbour is taken when its mask matches the edge frame (cheap, the
    usual case), or — when a moving background makes the masks differ but
    the scene is plausibly the same (mask similarity >= 50) — when reading
    that one frame gives the line's text.  Frames already owned by another
    line, blank frames and frames that read differently stop the growth."""
    if not lines:
        return 0
    owner = {}
    for i, ln in enumerate(lines):
        for f in range(ln.start_index, ln.end_index + 1):
            owner[f] = i
    blank = set()
    for g in groups:
        if g.blank:
            blank.update(range(g.start, g.end + 1))
    cache: dict[int, np.ndarray | None] = {}

    def mask_of(index: int):
        if index not in cache:
            cache[index] = make_mask(imread(files[index]), brightness_min)
        return cache[index]

    def same_text(index: int, ln: Line) -> bool:
        img = imread(files[index])
        if img is None:
            return False
        if use_mask and brightness_min > 0:
            img = masked_copy(img, brightness_min)
        text, _ = assemble_text(filter_boxes(ocr(img, min_score), ignore), lang)
        return bool(text) and text_similarity(text, ln.text) >= text_sim

    def takes(ref, idx: int, ln: Line) -> bool:
        m = mask_of(idx)
        if m is None or ref is None or m.shape != ref.shape:
            return False
        if similar(ref, m, brightness_min, image_sim):
            return True
        score = mask_similarity(ref, m) if brightness_min > 0 else gray_similarity(ref, m)
        return score >= 20 and same_text(idx, ln)

    step = 1000.0 / fps
    grown = 0
    for i, ln in enumerate(lines):
        ref = mask_of(ln.start_index)
        idx = ln.start_index - 1
        while idx >= 0 and idx not in blank and owner.get(idx) is None and takes(ref, idx, ln):
            owner[idx] = i
            ln.start_index, ln.start_ms = idx, frame_times[idx]
            ref = mask_of(idx)
            grown += 1
            idx -= 1
        ref = mask_of(ln.end_index)
        idx = ln.end_index + 1
        while idx < len(files) and idx not in blank and owner.get(idx) is None and takes(ref, idx, ln):
            owner[idx] = i
            ln.end_index = idx
            ln.end_ms = frame_times[idx + 1] if idx + 1 < len(frame_times) else frame_times[idx] + step
            ref = mask_of(idx)
            grown += 1
            idx += 1
    return grown


# ---------------------------------------------------------------- 5. refine
def _decode_window(video: str, start_ms: float, length_ms: float, crop_vf: str,
                   size: tuple[int, int]) -> list[tuple[float, np.ndarray]]:
    """Every source frame in [start, start+length) as (time_ms, BGR image).
    Input seeking re-bases timestamps at the seek point, so a frame's time is
    start_ms + its showinfo pts; the caller snaps start_ms onto the native
    frame grid so the first decoded frame really is at start_ms."""
    w, h = size
    # Seek a hair before the grid point: a seek string rounded to 4 decimals can
    # land a fraction of a tick *after* the frame's own timestamp, and accurate
    # seeking then drops that frame.  ffmpeg re-bases output timestamps at the
    # seek target, so the frame at start_ms comes back with pts_time = 2 ms and
    # the labels below stay exact.
    # -fps_mode passthrough: one output frame per source frame.  Left to its default,
    # ffmpeg sometimes duplicated a frame into a dropped-frame gap here, which shifted
    # every later label by one frame.
    seek_ms = max(0.0, start_ms - 2.0)
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-y",
           "-ss", f"{seek_ms / 1000.0:.4f}", "-i", video, "-t", f"{(length_ms + 2.0) / 1000.0:.3f}",
           "-vf", f"showinfo,{crop_vf}", *fps_mode_args(),
           "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    p = subprocess.run(cmd, capture_output=True)
    times = _parse_showinfo(p.stderr)
    frame_bytes = w * h * 3
    n = len(p.stdout) // frame_bytes
    frames = []
    for i in range(n):
        arr = np.frombuffer(p.stdout, dtype=np.uint8, count=frame_bytes, offset=i * frame_bytes)
        t = seek_ms + times[i] if i in times else start_ms + i * REFINE_FALLBACK_STEP_MS
        frames.append((t, arr.reshape(h, w, 3)))
    return frames


def _pick_transition(flags: list[tuple[float, bool]], find_start: bool):
    if find_start:
        for t, ok in flags:
            if ok:
                return t
        return None
    for i in range(len(flags) - 1, -1, -1):
        if flags[i][1]:
            return flags[i + 1][0] if i + 1 < len(flags) else flags[i][0] + REFINE_FALLBACK_STEP_MS
    return None


def refine_line(line: Line, video: str, frames_dir: Path, step_ms: float, frame_ms: float,
                brightness_min: int, image_sim: int, crop_vf: str, size: tuple[int, int]) -> None:
    def snap(ms: float) -> float:       # onto the native frame grid, never later than ms
        return max(0.0, math.floor(ms / frame_ms + 1e-6) * frame_ms)

    def coarse_mask(index: int, allow_blank: bool = False):
        f = frames_dir / f"img{index + 1:06d}.jpg"      # ffmpeg numbers from 1
        if index < 0 or not f.exists():
            return None
        m = make_mask(imread(str(f)), brightness_min)
        if m is None or (not allow_blank and brightness_min > 0 and is_blank(m)):
            return None
        return m

    def transition(frames, ref, other, find_start: bool):
        """Where inside the window the subtitle appears (find_start) or is gone.
        Primary rule: the frame pair with the largest jump of similarity to
        the coarse frame that has the subtitle — a bright, moving background
        lowers every similarity, but the text arriving or leaving is still the
        biggest single change.  Fallback (no clear jump): a frame shows the
        subtitle when it looks more like that reference than like the
        neighbouring coarse frame that does not have it."""
        sim = mask_similarity if brightness_min > 0 else gray_similarity
        sims = []
        for t, img in frames:
            m = make_mask(img, brightness_min)
            if m is None or m.shape != ref.shape or (brightness_min > 0 and is_blank(m)):
                sims.append((t, 0, 0))
                continue
            s_other = sim(other, m) if other is not None and other.shape == m.shape else -1
            sims.append((t, sim(ref, m), s_other))
        if not sims:
            return None
        best_i, best_jump = None, 0
        for i in range(1, len(sims)):
            jump = sims[i][1] - sims[i - 1][1]
            if not find_start:
                jump = -jump
            if jump > best_jump:
                best_i, best_jump = i, jump
        if best_i is not None and best_jump >= REFINE_MIN_JUMP:
            return sims[best_i][0]          # first frame with the text / first frame without it
        flags = [(t, s_ref >= REFINE_MIN_SIM and s_ref > s_other) for t, s_ref, s_other in sims]
        return _pick_transition(flags, find_start)

    ref = coarse_mask(line.start_index)
    if ref is not None:
        w0 = snap(line.start_ms - step_ms)
        frames = _decode_window(video, w0, line.start_ms - w0 + REFINE_PAD_MS, crop_vf, size)
        t = transition(frames, ref, coarse_mask(line.start_index - 1, True), True)
        if t is not None and t < line.start_ms:
            line.start_ms = t

    ref = coarse_mask(line.end_index)
    if ref is not None:
        w0 = snap(line.end_ms - step_ms)
        frames = _decode_window(video, w0, line.end_ms - w0 + REFINE_PAD_MS, crop_vf, size)
        t = transition(frames, ref, coarse_mask(line.end_index + 1, True), False)
        if t is not None and line.start_ms < t < line.end_ms:
            line.end_ms = t


def refine_lines(lines: list[Line], video: str, frames_dir: Path, coarse_fps: float,
                 native_fps: float, brightness_min: int, image_sim: int, crop_vf: str,
                 workers: int, progress=None) -> None:
    if not lines or coarse_fps >= 25:
        return
    sample = imread(str(next(frames_dir.glob("img*.jpg"))))
    size = (sample.shape[1], sample.shape[0])
    step_ms = 1000.0 / coarse_fps
    frame_ms = 1000.0 / (native_fps or 25.0)
    done = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(refine_line, ln, video, frames_dir, step_ms, frame_ms, brightness_min,
                          image_sim, crop_vf, size) for ln in lines]
        for f in cf.as_completed(futs):
            f.result()
            done += 1
            if progress:
                progress(done, len(lines))
    for i in range(1, len(lines)):              # refined boundaries must not overlap
        if lines[i].start_ms < lines[i - 1].end_ms:
            lines[i].start_ms = lines[i - 1].end_ms


# ---------------------------------------------------------------- 6. output
def fmt_ts(ms: float) -> str:
    ms = int(round(ms))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(lines: list[Line], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for n, ln in enumerate(lines, 1):
            f.write(f"{n}\n{fmt_ts(ln.start_ms)} --> {fmt_ts(ln.end_ms)}\n{ln.text}\n\n")


# ---------------------------------------------------------------- auto-crop
def _imwrite(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if ok:
        path.write_bytes(buf.tobytes())


def detect_subtitle_band(video: str, info: dict, ocr: "Ocr", scan_dir: Path, *,
                         search=DEFAULTS["search"], brightness_min: int = DEFAULTS["brightness_min"],
                         max_frames: int = 60) -> tuple[tuple, dict]:
    """Where do the subtitles sit?  Samples the search region (default: the
    bottom 60% of the frame) about once a second, runs text *detection* only
    on the brightness-masked samples, keeps boxes centred on the frame's
    vertical axis — subtitles are centred, scene text, logos and corner
    disclaimers are not — and takes the most populated row of them (plus the
    row above/below it for two-line subtitles) as the scan band.

    Returns (crop x%,y%,w%,h%, stats).  Falls back to the default bottom third
    when too little centred text was found."""
    W, H = info["width"], info["height"]
    duration = max(1.0, info["duration"])
    # 640 px wide is plenty for *finding* text (a 1080p subtitle glyph is still
    # ~25 px) and keeps the detector at ~0.3 s per frame.
    search_vf, (sx, sy, sw, sh) = crop_filter(info, search, 640)
    scale = min(1.0, 640.0 / sw)
    stats = dict(frames=0, boxes=0, centred=0)
    fallback = (tuple(DEFAULTS["crop"]), stats)
    boxes = []       # (cy, y0, y1, h, frame index) in full-frame pixels
    files = []
    # A video with sparse dialogue can show no subtitle on most of 60 samples,
    # so a thin first pass is followed by a four-times denser one.
    for attempt_frames in (max_frames, 4 * max_frames):
        shutil.rmtree(scan_dir, ignore_errors=True)
        scan_dir.mkdir(parents=True, exist_ok=True)
        interval = max(0.25, duration / attempt_frames)
        select = f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{interval:.3f})'"
        pattern = str(scan_dir / "scan%05d.jpg")
        cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", video,
               "-vf", f"{select},{search_vf}", *fps_mode_args(), "-q:v", "3", pattern]
        subprocess.run(cmd, capture_output=True)
        files = sorted(scan_dir.glob("scan*.jpg"))
        if not files:
            return fallback
        stats.update(frames=len(files), boxes=0, centred=0)
        boxes = []
        for i, f in enumerate(files):
            img = imread(f)
            if img is None:
                continue
            if brightness_min > 0:
                img = masked_copy(img, brightness_min)
            # Recognise, not just detect: bright water, skin or fabric makes the
            # detector fire on wide blobs, and they would then vote for a band.
            # Only boxes that read as at least two real characters count.
            for b in filter_boxes(ocr(img, 0.7), []):
                stats["boxes"] += 1
                if _content_len(b["txt"]) < 2:
                    continue
                fx0, fx1 = sx + b["x0"] / scale, sx + b["x1"] / scale
                fy0, fy1 = sy + b["y0"] / scale, sy + b["y1"] / scale
                h = fy1 - fy0
                if h < 0.008 * H or h > 0.12 * H:          # specks and huge blobs
                    continue
                if abs((fx0 + fx1) / 2 - W / 2) > 0.15 * W:   # not centred: not a subtitle
                    continue
                stats["centred"] += 1
                boxes.append(((fy0 + fy1) / 2, fy0, fy1, h, i))
        if len(boxes) >= 6 or interval <= 0.25:
            break
    if len(boxes) < 6:
        log.warning("auto-crop: only %d centred text boxes in %d frames — using the default scan area",
                    len(boxes), len(files))
        return fallback

    med_h = statistics.median(b[3] for b in boxes)
    nbins = 100
    hist = [0] * nbins
    for cy, *_ in boxes:
        hist[min(nbins - 1, int(cy / H * nbins))] += 1
    smooth = [sum(hist[max(0, i - 1):i + 2]) for i in range(nbins)]
    mode = max(range(nbins), key=lambda i: smooth[i])
    mode_cy = (mode + 0.5) * H / nbins
    core = [b for b in boxes if abs(b[0] - mode_cy) <= 2.5 * med_h]
    y0 = min(b[1] for b in core) - 0.75 * med_h
    y1 = max(b[2] for b in core) + 0.75 * med_h
    if y1 - y0 < 3.2 * med_h:                          # room for a two-line subtitle
        mid = (y0 + y1) / 2
        y0, y1 = mid - 1.6 * med_h, mid + 1.6 * med_h
    y0 = max(0.0, math.floor(y0)); y1 = min(float(H), math.ceil(y1))
    crop = (0.0, 100.0 * y0 / H, 100.0, 100.0 * (y1 - y0) / H)
    stats.update(band_px=(int(y0), int(y1)), line_height_px=int(med_h), core_boxes=len(core),
                 share_in_band=round(len(core) / len(boxes), 2))
    return crop, stats


def write_scan_preview(video: str, t_seconds: float, scan_px: tuple, path: Path,
                       search_px: tuple | None = None) -> bool:
    """One full frame at t with the scan area drawn on it (green) — and the
    auto-crop search region (orange) — so the band can be checked at a glance."""
    q = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", f"{t_seconds:.3f}", "-i", video, "-frames:v", "1", "-f", "image2pipe",
                        "-vcodec", "png", "pipe:1"], capture_output=True)
    frame = cv2.imdecode(np.frombuffer(q.stdout, np.uint8), cv2.IMREAD_COLOR) if q.stdout else None
    if frame is None:
        return False
    W = frame.shape[1]
    if search_px:
        sx, sy, sw, sh = search_px
        cv2.rectangle(frame, (sx, sy), (sx + sw - 1, sy + sh - 1), (0, 200, 255), max(1, W // 800))
    x, y, w, h = scan_px
    cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), max(2, W // 400))
    if W > 720:
        frame = cv2.resize(frame, (720, int(frame.shape[0] * 720 / W)), interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    _imwrite(path, frame)
    return True


# ---------------------------------------------------------------- driver
def run(video: str, output: str | None = None, *, lang: str = "ch", fps: int = DEFAULTS["fps"],
        crop=None, max_width: int = DEFAULTS["max_width"], auto_crop: bool = DEFAULTS["auto_crop"],
        search=DEFAULTS["search"], scan_preview: str | None = None,
        brightness_min: int = DEFAULTS["brightness_min"], image_sim: int = DEFAULTS["image_sim"],
        text_sim: int = DEFAULTS["text_sim"], max_gap: int = DEFAULTS["max_gap"],
        min_duration: int = DEFAULTS["min_duration"], bridge_gap: int = DEFAULTS["bridge_gap"],
        refine: bool = True, use_mask: bool = True, work_dir: str | None = None,
        keep_frames: bool = False, det: str = DEFAULTS["det"], rec: str = DEFAULTS["rec"],
        det_limit: int = DEFAULTS["det_limit"], unclip: float = DEFAULTS["unclip"],
        min_score: float = DEFAULTS["min_score"], static_share: float = DEFAULTS["static_share"],
        ignore_text: list[str] | None = None, gpu: bool = False, threads: int | None = None,
        side_text: bool = False, workers: int | None = None, ocr: "Ocr | None" = None,
        write_json: bool = True, progress=None) -> dict:
    t_all = time.time()
    video = str(Path(video).resolve())
    out_path = Path(output) if output else Path(video).with_suffix(".srt")
    stem = Path(video).stem
    work_root = Path(work_dir) if work_dir else Path(tempfile.gettempdir()) / "hardsub-ocr"
    # ASCII-only scratch folder: ffmpeg's %06d output pattern and OpenCV both
    # misbehave on non-ASCII paths on Windows, and video names are often CJK.
    job_dir = work_root / (re.sub(r"[^A-Za-z0-9._-]+", "_", stem)[:40] + "_" + hashlib.md5(video.encode("utf-8")).hexdigest()[:8])
    frames_dir = job_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir, ignore_errors=True)
    workers = workers or max(2, (os.cpu_count() or 4) // 2)
    ignore = [re.compile(p, re.IGNORECASE) for p in (ignore_text or [])]
    timings = {}

    info = probe(video)
    ocr = ocr or Ocr(lang, det, rec, det_limit, unclip, gpu, threads)
    band_stats = {}
    if crop is None and auto_crop:
        t = time.time()
        crop, band_stats = detect_subtitle_band(video, info, ocr, job_dir / "scan", search=search,
                                                brightness_min=brightness_min)
        timings["auto_crop_s"] = round(time.time() - t, 1)
        log.info("auto-crop: %s (%.1fs)", band_stats, timings["auto_crop_s"])
    elif crop is None:
        crop = DEFAULTS["crop"]
    crop_vf, crop_px = crop_filter(info, crop, max_width)
    log.info("%s: %dx%d %.3f fps %.1fs — scan area x,y,w,h=%s", stem, info["width"], info["height"],
             info["fps"], info["duration"], crop_px)

    t = time.time()
    files, frame_times = extract_frames(video, frames_dir, fps, crop_vf)
    timings["extract_s"] = round(time.time() - t, 1)
    log.info("%d frames sampled at %d fps (%.1fs)", len(files), fps, timings["extract_s"])

    t = time.time()
    groups = group_frames(files, frame_times, fps, brightness_min, image_sim)
    timings["group_s"] = round(time.time() - t, 1)
    n_text = sum(1 for g in groups if not g.blank)
    log.info("%d groups, %d with bright pixels (%.1fs)", len(groups), n_text, timings["group_s"])

    t = time.time()
    log.info("OCR models: %s", ocr.models)
    ocr_groups(groups, ocr, brightness_min, use_mask, min_score, ignore, side_text,
               progress=(lambda n, m: progress("ocr", n, m)) if progress else None)
    static = remove_static_text(groups, static_share, len(files))
    if static:
        log.info("static text removed (watermark/logo): %s", static)
    for g in groups:
        g.text, g.confidence = assemble_text(g.boxes, ocr.lang)
    timings["ocr_s"] = round(time.time() - t, 1)
    log.info("OCR of %d frames done, %d with text (%.1fs)", n_text,
             sum(1 for g in groups if g.text), timings["ocr_s"])

    lines = build_lines(groups, text_sim, max_gap, min_duration)
    lines = filter_lines(lines, min_duration)
    lines = bridge_lines(lines, text_sim, max_gap, bridge_gap)
    grown = extend_lines(lines, groups, files, frame_times, fps, brightness_min, image_sim,
                         text_sim, ocr, use_mask, min_score, ignore, ocr.lang)
    log.info("%d subtitle lines (edges grown over %d unread frames)", len(lines), grown)

    if refine:
        t = time.time()
        refine_lines(lines, video, frames_dir, fps, info["fps"], brightness_min, image_sim, crop_vf,
                     workers, progress=(lambda n, m: progress("refine", n, m)) if progress else None)
        timings["refine_s"] = round(time.time() - t, 1)
        log.info("timing refined at native frame rate (%.1fs)", timings["refine_s"])

    lines = [ln for ln in lines if ln.text.strip()]
    if scan_preview:
        # the frame of the longest confident cue: a subtitle is guaranteed to be in view
        preview = Path(scan_preview) / (stem + ".scan.png")
        pick = max(lines, key=lambda ln: (ln.confidence >= 0.9, len(ln.text), ln.end_ms - ln.start_ms), default=None)
        t_prev = ((pick.start_ms + pick.end_ms) / 2000.0) if pick else min(5.0, info["duration"] / 2)
        search_px = crop_filter(info, search, 0)[1] if band_stats else None
        if write_scan_preview(video, t_prev, crop_px, preview, search_px):
            band_stats["preview"] = str(preview)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_srt(lines, out_path)
    timings["total_s"] = round(time.time() - t_all, 1)
    result = {
        "video": video, "srt": str(out_path), "cues": len(lines),
        "duration_s": info["duration"], "native_fps": info["fps"], "scan_area_px": crop_px,
        "scan_area_pct": [round(c, 2) for c in crop], "auto_crop": band_stats,
        "settings": dict(lang=lang, models=ocr.models, fps=fps, crop=[round(c, 2) for c in crop],
                         auto_crop=auto_crop and not band_stats == {}, max_width=max_width,
                         brightness_min=brightness_min, image_sim=image_sim, text_sim=text_sim,
                         max_gap=max_gap, min_duration=min_duration, bridge_gap=bridge_gap,
                         refine=refine, mask=use_mask, det=det, rec=rec, det_limit=det_limit,
                         unclip=unclip, min_score=min_score, static_share=static_share,
                         ignore_text=ignore_text or [], gpu=gpu, threads=threads,
                         side_text=side_text),
        "static_text_removed": static,
        "timings": timings,
        "low_confidence": sum(1 for ln in lines if ln.confidence < 0.8),
        "lines": [dict(index=i, start_ms=round(ln.start_ms, 1), end_ms=round(ln.end_ms, 1),
                       text=ln.text, confidence=round(ln.confidence, 3))
                  for i, ln in enumerate(lines, 1)],
    }
    if write_json:
        with open(out_path.with_suffix(".ocr.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
    if not keep_frames:
        shutil.rmtree(job_dir, ignore_errors=True)
    log.info("wrote %s (%d cues, %.1fs total)", out_path, len(lines), timings["total_s"])
    return result


def parse_crop(s: str):
    parts = [float(p) for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop must be x%,y%,w%,h% e.g. 0,66.7,100,33.3")
    return tuple(parts)


def add_ocr_args(p: argparse.ArgumentParser) -> None:
    """OCR options shared by hardsub_ocr.py and batch_hardsub.py."""
    p.add_argument("--lang", default="ch",
                   help="subtitle language: ch (Chinese+English), en, vi/latin (Vietnamese, French, ...), "
                        "ko/korean, ja/japan, zh-tw/chinese_cht, ru/cyrillic, ar/arabic, th ... (default: ch)")
    p.add_argument("--fps", type=int, default=DEFAULTS["fps"], help="scan frames per second (default 5)")
    p.add_argument("--crop", type=parse_crop, default=None,
                   help="scan area as x%%,y%%,w%%,h%% of the frame; default: found automatically "
                        "(--no-auto-crop: the bottom third, 0,66.7,100,33.3)")
    p.add_argument("--no-auto-crop", action="store_true",
                   help="skip the subtitle-band detection and scan the bottom third")
    p.add_argument("--search", type=parse_crop, default=DEFAULTS["search"],
                   help="where auto-crop looks for subtitles, x%%,y%%,w%%,h%% (default 0,40,100,60 = bottom 60%%)")
    p.add_argument("--scan-preview", metavar="DIR",
                   help="write <video>.scan.png here showing the chosen scan band on a subtitle frame")
    p.add_argument("--max-width", type=int, default=DEFAULTS["max_width"])
    p.add_argument("--brightness-min", type=int, default=DEFAULTS["brightness_min"],
                   help="pixels below this luma (0-255) are not subtitle text; 0 disables masking (default 190)")
    p.add_argument("--image-sim", type=int, default=DEFAULTS["image_sim"], help="frame grouping similarity %% (92)")
    p.add_argument("--text-sim", type=int, default=DEFAULTS["text_sim"], help="line merge text similarity %% (80)")
    p.add_argument("--max-gap", type=int, default=DEFAULTS["max_gap"], help="ms (250)")
    p.add_argument("--min-duration", type=int, default=DEFAULTS["min_duration"], help="ms (250)")
    p.add_argument("--bridge-gap", type=int, default=DEFAULTS["bridge_gap"],
                   help="re-join same-text lines split by an OCR miss up to this gap in ms (1000)")
    p.add_argument("--no-refine", action="store_true", help="skip frame-accurate boundary refinement")
    p.add_argument("--no-mask", action="store_true", help="OCR the natural frame instead of the masked copy")
    p.add_argument("--det", default=DEFAULTS["det"],
                   help="detector: v6-small (default), v6-medium, v5-mobile, v5-server, v4-mobile")
    p.add_argument("--rec", default=DEFAULTS["rec"],
                   help="recogniser: auto (default, per language), mobile, server, or e.g. v5-server")
    p.add_argument("--det-limit", type=int, default=DEFAULTS["det_limit"],
                   help="detector input: longer side is downscaled to at most this (960)")
    p.add_argument("--unclip", type=float, default=DEFAULTS["unclip"], help="detector box expansion (2.2)")
    p.add_argument("--min-score", type=float, default=DEFAULTS["min_score"], help="drop OCR boxes below (0.5)")
    p.add_argument("--static-share", type=float, default=DEFAULTS["static_share"],
                   help="watermark rule: text at one position in >= this share of frames (0.4; 0 = off)")
    p.add_argument("--ignore-text", action="append", default=[], metavar="REGEX",
                   help="drop OCR boxes matching this pattern (repeatable), e.g. a channel name")
    p.add_argument("--gpu", action="store_true", help="run ONNX Runtime on CUDA (needs onnxruntime-gpu)")
    p.add_argument("--threads", type=int, help="CPU threads for the OCR engine (default: all cores)")
    p.add_argument("--keep-side-text", action="store_true",
                   help="keep 1-4 character text far from the centre (end cards, side logos are dropped by default)")
    p.add_argument("--work-dir", help="where sampled frames go (default: system temp)")
    p.add_argument("--keep-frames", action="store_true")
    p.add_argument("--workers", type=int, help="parallel ffmpeg calls during refinement")


def ocr_kwargs(args: argparse.Namespace) -> dict:
    return dict(lang=args.lang, fps=args.fps, crop=args.crop, auto_crop=not args.no_auto_crop,
                search=args.search, scan_preview=args.scan_preview, max_width=args.max_width,
                brightness_min=args.brightness_min, image_sim=args.image_sim, text_sim=args.text_sim,
                max_gap=args.max_gap, min_duration=args.min_duration, bridge_gap=args.bridge_gap,
                refine=not args.no_refine, use_mask=not args.no_mask, det=args.det, rec=args.rec,
                det_limit=args.det_limit, unclip=args.unclip, min_score=args.min_score,
                static_share=args.static_share, ignore_text=args.ignore_text, gpu=args.gpu,
                threads=args.threads, side_text=args.keep_side_text, work_dir=args.work_dir,
                keep_frames=args.keep_frames, workers=args.workers)


def main(argv=None):
    p = argparse.ArgumentParser(description="OCR burned-in subtitles from a video into .srt")
    p.add_argument("video")
    p.add_argument("-o", "--output", help="output .srt (default: next to the video)")
    add_ocr_args(p)
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    result = run(args.video, args.output, **ocr_kwargs(args))
    print(f"{result['cues']} cues -> {result['srt']}  ({result['timings']['total_s']}s)")


if __name__ == "__main__":
    main()
