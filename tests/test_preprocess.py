"""Tests for the preprocess module."""

from __future__ import annotations

import numpy as np
import pytest

import config
import preprocess


@pytest.fixture
def synthetic_frame() -> np.ndarray:
    """A deterministic 480x640x3 BGR frame."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)


def test_resize_returns_correct_shape(synthetic_frame: np.ndarray) -> None:
    resized = preprocess.resize_frame(synthetic_frame, 320, 240)
    assert resized.shape == (240, 320, 3)


def test_grayscale_returns_single_channel(synthetic_frame: np.ndarray) -> None:
    gray = preprocess.to_grayscale(synthetic_frame)
    assert gray.ndim == 2
    assert gray.shape == (480, 640)


def test_clahe_does_not_crash(synthetic_frame: np.ndarray) -> None:
    gray = preprocess.to_grayscale(synthetic_frame)
    enhanced = preprocess.apply_clahe(gray)
    assert enhanced.shape == gray.shape
    assert enhanced.dtype == np.uint8


def test_augment_returns_minimum_six_variants(synthetic_frame: np.ndarray) -> None:
    variants = preprocess.augment_image(synthetic_frame)
    assert len(variants) >= 6
    for v in variants:
        assert isinstance(v, np.ndarray)
        assert v.shape == synthetic_frame.shape


def test_pipeline_returns_tuple_of_two_frames(synthetic_frame: np.ndarray) -> None:
    color, gray = preprocess.full_preprocess_pipeline(synthetic_frame)
    assert color.shape == (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3)
    assert gray.shape == (config.FRAME_HEIGHT, config.FRAME_WIDTH)
    assert color.dtype == np.uint8
    assert gray.dtype == np.uint8
