"""Unit tests for the validation metric functions.

These cover the pure-math core (genuine/impostor split, FAR/FRR sweep,
EER) using synthetic distances, so they run fast and need neither the
``face_recognition`` model nor any dataset download.
"""

from __future__ import annotations

import numpy as np
import pytest

import validate


def test_genuine_impostor_split_counts():
    # 2 identities, 2 samples each -> C(4,2)=6 pairs: 2 genuine, 4 impostor
    embeddings = np.array([[0.0], [0.1], [10.0], [10.1]])
    labels = np.array(["a", "a", "b", "b"], dtype=object)
    genuine, impostor = validate.genuine_impostor_distances(embeddings, labels)
    assert genuine.size == 2
    assert impostor.size == 4
    # genuine pairs are close, impostor pairs are far
    assert genuine.max() < impostor.min()


def test_threshold_sweep_monotonic_far():
    genuine = np.array([0.1, 0.2, 0.15])
    impostor = np.array([0.8, 0.9, 0.85])
    sweep = validate.threshold_sweep(genuine, impostor,
                                     thresholds=np.linspace(0, 1, 11))
    # FAR is non-decreasing as the threshold grows
    far = sweep["FAR"].to_numpy()
    assert np.all(np.diff(far) >= -1e-9)
    # FRR is non-increasing as the threshold grows
    frr = sweep["FRR"].to_numpy()
    assert np.all(np.diff(frr) <= 1e-9)


def test_threshold_sweep_extremes():
    genuine = np.array([0.1, 0.2])
    impostor = np.array([0.8, 0.9])
    sweep = validate.threshold_sweep(genuine, impostor,
                                     thresholds=np.array([0.0, 0.5, 1.0]))
    # threshold 0: reject everything -> FAR 0, FRR 1
    row0 = sweep.iloc[0]
    assert row0["FAR"] == pytest.approx(0.0)
    assert row0["FRR"] == pytest.approx(1.0)
    # threshold 1: accept everything -> FAR 1, FRR 0
    row_last = sweep.iloc[-1]
    assert row_last["FAR"] == pytest.approx(1.0)
    assert row_last["FRR"] == pytest.approx(0.0)


def test_compute_eer_well_separated():
    # Perfectly separable: genuine << impostor -> EER near 0
    genuine = np.random.default_rng(0).normal(0.2, 0.02, 200)
    impostor = np.random.default_rng(1).normal(0.9, 0.02, 200)
    sweep = validate.threshold_sweep(genuine, impostor)
    eer, eer_t = validate.compute_eer(sweep)
    assert eer < 0.05
    assert 0.2 < eer_t < 0.9


def test_optimal_threshold_between_clusters():
    genuine = np.array([0.1, 0.15, 0.2])
    impostor = np.array([0.7, 0.8, 0.9])
    sweep = validate.threshold_sweep(genuine, impostor)
    t = validate.optimal_threshold(sweep)
    assert 0.2 <= t <= 0.7


def test_synthetic_dataset_is_separable_and_runs():
    ds = validate.make_synthetic_dataset(n_identities=4, per_identity=8)
    assert ds.embeddings is not None
    assert ds.n_images == 32
    assert len(ds.identities) == 4
    genuine, impostor = validate.genuine_impostor_distances(ds.embeddings, ds.labels)
    # clustered embeddings -> genuine pairs closer on average than impostor
    assert genuine.mean() < impostor.mean()


def test_identification_cv_on_synthetic():
    ds = validate.make_synthetic_dataset(n_identities=4, per_identity=10)
    # threshold 0.6 sits between the ~0.35 genuine and ~0.85 impostor scale
    res = validate.identification_cv(ds.embeddings, ds.labels,
                                     threshold=0.6, n_splits=5)
    assert 0.0 <= res["accuracy"] <= 1.0
    assert res["confusion_matrix"].shape[0] == res["confusion_matrix"].shape[1]
    # well-separated synthetic clusters should classify nearly perfectly
    assert res["accuracy"] > 0.8
