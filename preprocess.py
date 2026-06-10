"""Image preprocessing helpers.

Functions take and return numpy arrays in OpenCV BGR layout unless noted.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from utils import get_logger

logger = get_logger(__name__)


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize ``frame`` to exactly ``(width, height)`` pixels.

    Args:
        frame: Input image of shape (H, W) or (H, W, C).
        width: Target width in pixels.
        height: Target height in pixels.

    Returns:
        The resized image with shape (height, width[, C]).
    """
    if frame is None:
        raise ValueError("frame is None")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    return cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR (or BGRA) image to single-channel grayscale.

    If ``frame`` is already 2D it is returned unchanged.
    """
    if frame is None:
        raise ValueError("frame is None")
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray_frame: np.ndarray) -> np.ndarray:
    """Apply CLAHE with ``clipLimit=2.0`` and ``tileGridSize=(8, 8)``.

    Args:
        gray_frame: Single-channel uint8 image.

    Returns:
        Contrast-enhanced single-channel uint8 image.
    """
    if gray_frame is None:
        raise ValueError("gray_frame is None")
    if gray_frame.ndim != 2:
        gray_frame = to_grayscale(gray_frame)
    if gray_frame.dtype != np.uint8:
        gray_frame = gray_frame.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_frame)


def denoise(frame: np.ndarray) -> np.ndarray:
    """Smooth ``frame`` with a 5x5 Gaussian blur.

    Args:
        frame: Image of any channel count.

    Returns:
        Blurred image with the same shape and dtype.
    """
    if frame is None:
        raise ValueError("frame is None")
    return cv2.GaussianBlur(frame, (5, 5), 0)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Scale pixel values to ``[0, 1]`` and return as uint8 in ``[0, 255]``.

    The temporary float32 conversion handles inputs whose intensity range
    is not the full 0-255 span (e.g., dim camera frames).
    """
    if frame is None:
        raise ValueError("frame is None")
    f = frame.astype(np.float32)
    min_v = float(f.min())
    max_v = float(f.max())
    if max_v - min_v < 1e-6:
        return np.zeros_like(frame, dtype=np.uint8)
    scaled = (f - min_v) / (max_v - min_v)
    return (scaled * 255.0).astype(np.uint8)


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def _adjust_brightness(image: np.ndarray, delta: int) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=1.0, beta=float(delta))


def _add_gaussian_noise(image: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def augment_image(image: np.ndarray) -> List[np.ndarray]:
    """Return a list of augmented versions of ``image``.

    The output always contains at least 6 variants:
        1. original
        2. horizontal flip
        3. rotation by +15 degrees
        4. rotation by -15 degrees
        5. brightness +40
        6. brightness -40
        7. gaussian noise injection
    """
    if image is None:
        raise ValueError("image is None")

    variants: List[np.ndarray] = [
        image.copy(),
        cv2.flip(image, 1),
        _rotate(image, 15),
        _rotate(image, -15),
        _adjust_brightness(image, 40),
        _adjust_brightness(image, -40),
        _add_gaussian_noise(image),
    ]
    logger.debug("augment_image produced %d variants", len(variants))
    return variants


def full_preprocess_pipeline(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run the standard preprocessing pipeline for detection.

    Steps:
        1. Resize to ``(FRAME_WIDTH, FRAME_HEIGHT)`` from config.
        2. Denoise (Gaussian 5x5).
        3. Convert to grayscale and apply CLAHE.

    Args:
        frame: Raw BGR camera frame.

    Returns:
        Tuple ``(color_frame, gray_enhanced_frame)`` where ``color_frame``
        is the resized + denoised BGR image and ``gray_enhanced_frame`` is
        the CLAHE-enhanced grayscale image.
    """
    import config  # local import keeps test overrides easy

    resized = resize_frame(frame, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    color_frame = denoise(resized)
    gray = to_grayscale(color_frame)
    gray_enhanced = apply_clahe(gray)
    return color_frame, gray_enhanced
