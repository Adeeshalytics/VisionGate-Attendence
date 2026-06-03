"""Configuration constants for VisionGate.

All paths use pathlib.Path. Values can be overridden via a .env file
loaded with python-dotenv.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT: Path = Path(__file__).resolve().parent


def _env(name: str, default):
    """Return env var cast to the type of default, or default if unset."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    if isinstance(default, bool):
        return raw.lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


# Camera / frame capture
CAMERA_INDEX: int = _env("CAMERA_INDEX", 0)
FRAME_WIDTH: int = _env("FRAME_WIDTH", 640)
FRAME_HEIGHT: int = _env("FRAME_HEIGHT", 480)
FRAME_SKIP: int = _env("FRAME_SKIP", 3)

# Enrollment / recognition
ENROLLMENT_IMAGES: int = _env("ENROLLMENT_IMAGES", 20)
RECOGNITION_THRESHOLD: float = _env("RECOGNITION_THRESHOLD", 0.45)
MIN_FACE_CONFIDENCE: float = _env("MIN_FACE_CONFIDENCE", 0.75)
MIN_FACE_SIZE: tuple[int, int] = (60, 60)

# Haar cascade bundled with OpenCV
HAAR_CASCADE_PATH: Path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"

# Data paths
DB_PATH: Path = PROJECT_ROOT / "data" / "attendance.db"
ENCODINGS_PATH: Path = PROJECT_ROOT / "data" / "encodings" / "encodings.pkl"
LBPH_MODEL_PATH: Path = PROJECT_ROOT / "data" / "encodings" / "lbph_model.yml"
ATTENDANCE_CSV_DIR: Path = PROJECT_ROOT / "data" / "attendance"
ENROLLED_FACES_DIR: Path = PROJECT_ROOT / "data" / "enrolled_faces"

# Attendance behavior
DUPLICATE_WINDOW_SECONDS: int = _env("DUPLICATE_WINDOW_SECONDS", 300)

# Liveness (eye aspect ratio)
EAR_THRESHOLD: float = _env("EAR_THRESHOLD", 0.25)
EAR_CONSEC_FRAMES: int = _env("EAR_CONSEC_FRAMES", 2)

# Logging
LOG_LEVEL: str = _env("LOG_LEVEL", "INFO")
