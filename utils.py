"""Shared utilities for VisionGate: logging, directory bootstrap, drawing helpers, session IDs."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

import config

_LOG_DIR: Path = config.PROJECT_ROOT / "logs"
_LOG_FILE: Path = _LOG_DIR / "visiongate.log"
_LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

_STATUS_COLORS: dict[str, tuple[int, int, int]] = {
    "recognized": (0, 200, 0),    # GREEN (BGR)
    "unknown": (0, 0, 220),       # RED
    "spoof": (0, 220, 220),       # YELLOW
}


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes to console and ``logs/visiongate.log``.

    The logger level is taken from ``config.LOG_LEVEL``. Handlers are only
    attached once per logger to avoid duplicate output on repeated calls.
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    level = getattr(logging, str(config.LOG_LEVEL).upper(), logging.INFO)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(_LOG_FORMAT)

        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(formatter)
        logger.addHandler(console)

        file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.propagate = False

    return logger


def ensure_dirs() -> None:
    """Create every data directory referenced in config if missing."""
    paths_to_create: list[Path] = [
        config.ENROLLED_FACES_DIR,
        config.ATTENDANCE_CSV_DIR,
        config.ENCODINGS_PATH.parent,
        config.LBPH_MODEL_PATH.parent,
        config.DB_PATH.parent,
        _LOG_DIR,
    ]
    for path in paths_to_create:
        path.mkdir(parents=True, exist_ok=True)


def draw_bounding_box(
    frame: np.ndarray,
    bbox: Sequence[int],
    label: str,
    confidence: float,
    status: str,
) -> np.ndarray:
    """Annotate ``frame`` with a coloured bounding box and label.

    Args:
        frame: BGR image to draw on (modified in-place and returned).
        bbox: Either (x, y, w, h) or (x1, y1, x2, y2).
        label: Text label (e.g., student name or "Unknown").
        confidence: Value in [0, 1] rendered as a percentage on the label.
        status: One of ``"recognized"``, ``"unknown"``, ``"spoof"``.
            Unknown statuses fall back to white.

    Returns:
        The annotated frame.
    """
    if len(bbox) != 4:
        raise ValueError("bbox must have exactly 4 values")

    a, b, c, d = (int(v) for v in bbox)
    # Heuristic: if 3rd/4th are smaller than 1st/2nd, assume (x, y, w, h)
    if c < a or d < b:
        x1, y1, x2, y2 = a, b, a + c, b + d
    else:
        x1, y1, x2, y2 = a, b, c, d

    color = _STATUS_COLORS.get(status, (255, 255, 255))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label} {confidence * 100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    label_y2 = y1
    label_y1 = max(0, y1 - th - baseline - 6)
    label_x2 = min(frame.shape[1], x1 + tw + 8)
    cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, thickness=-1)

    cv2.putText(
        frame,
        text,
        (x1 + 4, y1 - baseline - 2),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        lineType=cv2.LINE_AA,
    )

    return frame


def get_session_id(now: datetime | None = None) -> str:
    """Return a session identifier rounded to the nearest hour.

    Format: ``YYYY-MM-DD_HH-00``. Minutes >= 30 round up.
    """
    if now is None:
        now = datetime.now()
    if now.minute >= 30:
        now = now + timedelta(hours=1)
    rounded = now.replace(minute=0, second=0, microsecond=0)
    return rounded.strftime("%Y-%m-%d_%H-%M")
