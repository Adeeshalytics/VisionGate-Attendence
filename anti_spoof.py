"""Blink-based liveness check.

Uses dlib's 68-point landmarks to compute the Eye Aspect Ratio (EAR). A blink
is counted when the EAR dips below config.EAR_THRESHOLD for at least
config.EAR_CONSEC_FRAMES frames in a row. Within a rolling BLINK_WINDOW of
frames, we treat the subject as live once we've seen at least one blink -
which is enough to reject a held-up photo.
"""

from __future__ import annotations

import bz2
import urllib.request
from pathlib import Path
from typing import Sequence

import cv2
import dlib
import numpy as np

import config
import preprocess
from utils import get_logger

logger = get_logger(__name__)

# dlib 68-point predictor (downloaded on first use)
_LANDMARK_MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
_LANDMARK_MODEL_PATH: Path = (
    config.PROJECT_ROOT / "models" / "shape_predictor_68_face_landmarks.dat"
)

# 68-point indices for each eye (0-based)
LEFT_EYE_IDX: tuple[int, ...] = (36, 37, 38, 39, 40, 41)
RIGHT_EYE_IDX: tuple[int, ...] = (42, 43, 44, 45, 46, 47)


def get_ear(eye: Sequence[Sequence[float]]) -> float:
    """Compute the Eye Aspect Ratio for a single eye.

    Args:
        eye: Six (x, y) landmark coordinates, ordered along the eye
            contour as produced by dlib's 68-point predictor.

    Returns:
        The EAR value::

            EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

        Returns 0.0 if the horizontal distance is zero.
    """
    pts = np.asarray(eye, dtype=np.float64)
    if pts.shape != (6, 2):
        raise ValueError(f"eye must have shape (6, 2); got {pts.shape}")
    vertical_1 = np.linalg.norm(pts[1] - pts[5])
    vertical_2 = np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])
    if horizontal < 1e-6:
        return 0.0
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def _ensure_landmark_model() -> Path:
    """Download and bz2-extract the dlib 68-point predictor if missing."""
    _LANDMARK_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _LANDMARK_MODEL_PATH.exists():
        return _LANDMARK_MODEL_PATH

    bz2_path = _LANDMARK_MODEL_PATH.with_suffix(_LANDMARK_MODEL_PATH.suffix + ".bz2")
    logger.info("Downloading dlib 68-point landmark model to %s", bz2_path)
    urllib.request.urlretrieve(_LANDMARK_MODEL_URL, bz2_path)

    logger.info("Extracting %s", bz2_path)
    with bz2.open(bz2_path, "rb") as src, open(_LANDMARK_MODEL_PATH, "wb") as dst:
        for chunk in iter(lambda: src.read(1 << 16), b""):
            dst.write(chunk)
    bz2_path.unlink(missing_ok=True)
    return _LANDMARK_MODEL_PATH


class LivenessDetector:
    """Blink-based liveness gate driven by dlib facial landmarks.

    It's stateful: it tracks how many frames in a row the EAR has been below
    threshold and counts a blink when that run ends after at least
    config.EAR_CONSEC_FRAMES frames. A subject counts as live as long as at
    least one blink happened in the last BLINK_WINDOW frames.
    """

    BLINK_WINDOW: int = 60  # frames

    def __init__(self) -> None:
        model_path = _ensure_landmark_model()
        self._predictor = dlib.shape_predictor(str(model_path))
        self.blink_count: int = 0
        self.ear_below_threshold_frames: int = 0
        self.frame_counter: int = 0
        self._last_ear: float = 0.0
        self._was_live: bool = False

    def reset(self) -> None:
        """Clear all counters. Call at the start of a fresh session."""
        self.blink_count = 0
        self.ear_below_threshold_frames = 0
        self.frame_counter = 0
        self._last_ear = 0.0
        self._was_live = False

    @staticmethod
    def _landmarks_to_array(shape) -> np.ndarray:
        return np.array(
            [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)],
            dtype=np.float64,
        )

    def check(self, frame: np.ndarray, face_bbox: Sequence[int]) -> dict:
        """Update state with one frame and return the current liveness verdict.

        Args:
            frame: BGR or grayscale image containing the face.
            face_bbox: ``(x, y, w, h)`` of the detected face within ``frame``.

        Returns:
            A dict with keys ``is_live`` (bool), ``blink_count`` (int) and
            ``ear`` (float, last computed average EAR for the frame).
        """
        if frame is None:
            raise ValueError("frame is None")
        if len(face_bbox) != 4:
            raise ValueError("face_bbox must be (x, y, w, h)")

        gray = preprocess.to_grayscale(frame)
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)

        x, y, w, h = (int(v) for v in face_bbox)
        rect = dlib.rectangle(x, y, x + w, y + h)
        shape = self._predictor(gray, rect)
        landmarks = self._landmarks_to_array(shape)

        left_eye = landmarks[list(LEFT_EYE_IDX)]
        right_eye = landmarks[list(RIGHT_EYE_IDX)]
        ear = (get_ear(left_eye) + get_ear(right_eye)) / 2.0
        self._last_ear = ear

        if ear < config.EAR_THRESHOLD:
            self.ear_below_threshold_frames += 1
        else:
            if self.ear_below_threshold_frames >= config.EAR_CONSEC_FRAMES:
                self.blink_count += 1
                logger.debug(
                    "Blink #%d detected (EAR=%.3f)", self.blink_count, ear
                )
            self.ear_below_threshold_frames = 0

        self.frame_counter += 1
        is_live = self.blink_count > 0
        self._was_live = self._was_live or is_live

        if self.frame_counter >= self.BLINK_WINDOW:
            if self.blink_count == 0:
                logger.warning(
                    "No blink detected in last %d frames — possible spoof",
                    self.BLINK_WINDOW,
                )
            self.frame_counter = 0
            self.blink_count = 0
            self.ear_below_threshold_frames = 0

        return {"is_live": is_live, "blink_count": self.blink_count, "ear": ear}
