"""Evaluation suite for the face-recognition pipeline.

Produces the numbers and plots you'd want in the project report:
genuine vs. impostor distance distributions, FAR/FRR, an ROC curve with the
Equal Error Rate (which is what justifies our distance threshold rather than
just guessing it), identification accuracy/precision/recall/F1 with a
confusion matrix under stratified k-fold CV, and a dlib-vs-LBPH comparison.

Pick the data with --source:
    auto       local test set if present, else LFW (Olivetti fallback), plus
               any enrolled faces (default)
    lfw        Labeled Faces in the Wild subset (downloaded via sklearn)
    olivetti   Olivetti faces (tiny, works offline)
    enrolled   faces captured during enrollment
    local      images under data/test_faces/<identity>/
    synthetic  fabricated data - no download, no dlib, instant smoke test

    python validate.py                 # auto data, 5-fold CV
    python validate.py --source lfw
    python validate.py --source synthetic
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless: save figures without a display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import pairwise_distances

import config
from utils import get_logger

logger = get_logger(__name__)

VALIDATION_DIR: Path = config.PROJECT_ROOT / "data" / "validation"
LOCAL_TEST_DIR: Path = config.PROJECT_ROOT / "data" / "test_faces"
LBPH_FACE_SIZE = (200, 200)
UNKNOWN_LABEL = "__unknown__"

# Thresholds swept when computing FAR/FRR/ROC (face_recognition distance).
DEFAULT_THRESHOLDS = np.round(np.linspace(0.0, 1.0, 101), 3)


# Dataset container

@dataclass
class FaceDataset:
    """A labelled set of face images plus lazily-computed embeddings.

    Attributes:
        images_rgb: List of HxWx3 uint8 RGB images (for dlib encoding).
        images_gray: List of HxW uint8 grayscale images (for LBPH).
        labels: 1-D array of string identity labels, aligned with images.
        source: Human-readable description of where the data came from.
        embeddings: filled in by encode_dataset (N x 128 float).
    """

    images_rgb: list[np.ndarray] = field(default_factory=list)
    images_gray: list[np.ndarray] = field(default_factory=list)
    labels: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    source: str = "unknown"
    embeddings: np.ndarray | None = None

    @property
    def n_images(self) -> int:
        return len(self.images_rgb)

    @property
    def identities(self) -> list[str]:
        return sorted(set(self.labels.tolist()))


# Dataset loaders

def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
    return img


def _split_rgb_gray(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (rgb_uint8, gray_uint8) from any image."""
    img = _to_uint8(img)
    if img.ndim == 2:
        gray = img
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = img
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.ascontiguousarray(rgb), np.ascontiguousarray(gray)


def load_lfw_dataset(min_faces: int = 20, max_identities: int = 6,
                     max_per_identity: int = 15) -> FaceDataset:
    """Load a small Labeled Faces in the Wild subset via scikit-learn."""
    from sklearn.datasets import fetch_lfw_people

    logger.info("Fetching LFW (min_faces_per_person=%d) ...", min_faces)
    lfw = fetch_lfw_people(min_faces_per_person=min_faces, resize=0.5, color=True)
    names = lfw.target_names
    ds = FaceDataset(source=f"LFW (>= {min_faces} faces/person)")

    # Keep the identities with the most samples, capped for speed.
    counts = np.bincount(lfw.target)
    chosen = np.argsort(counts)[::-1][:max_identities]
    per_id: dict[int, int] = {c: 0 for c in chosen}

    labels: list[str] = []
    for img, tgt in zip(lfw.images, lfw.target):
        if tgt not in per_id or per_id[tgt] >= max_per_identity:
            continue
        rgb, gray = _split_rgb_gray(img)
        ds.images_rgb.append(rgb)
        ds.images_gray.append(gray)
        labels.append(str(names[tgt]))
        per_id[tgt] += 1
    ds.labels = np.array(labels, dtype=object)
    logger.info("LFW subset: %d images, %d identities", ds.n_images, len(ds.identities))
    return ds


def load_olivetti_dataset(max_identities: int = 10) -> FaceDataset:
    """Load the Olivetti faces dataset (40 people x 10 images, 64x64 gray)."""
    from sklearn.datasets import fetch_olivetti_faces

    logger.info("Fetching Olivetti faces ...")
    data = fetch_olivetti_faces()
    ds = FaceDataset(source="Olivetti faces")
    labels: list[str] = []
    for img, tgt in zip(data.images, data.target):
        if tgt >= max_identities:
            continue
        upscaled = cv2.resize(_to_uint8(img), (160, 160), interpolation=cv2.INTER_CUBIC)
        rgb, gray = _split_rgb_gray(upscaled)
        ds.images_rgb.append(rgb)
        ds.images_gray.append(gray)
        labels.append(f"olivetti_{tgt:02d}")
    ds.labels = np.array(labels, dtype=object)
    logger.info("Olivetti subset: %d images, %d identities", ds.n_images, len(ds.identities))
    return ds


def _load_from_dir(root: Path, source_name: str) -> FaceDataset:
    """Load images from ``root/<identity>/*.{png,jpg,jpeg}``."""
    ds = FaceDataset(source=source_name)
    labels: list[str] = []
    if not root.exists():
        return ds
    for identity_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img_path in sorted(identity_dir.iterdir()):
            if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb, gray = _split_rgb_gray(rgb)
            ds.images_rgb.append(rgb)
            ds.images_gray.append(gray)
            labels.append(identity_dir.name)
    ds.labels = np.array(labels, dtype=object)
    return ds


def load_enrolled_dataset() -> FaceDataset:
    """Load the face crops saved during enrollment."""
    ds = _load_from_dir(config.ENROLLED_FACES_DIR, "Enrolled faces")
    logger.info("Enrolled faces: %d images, %d identities", ds.n_images, len(ds.identities))
    return ds


def load_local_dataset() -> FaceDataset:
    """Load user-provided test images from data/test_faces/<identity>/."""
    ds = _load_from_dir(LOCAL_TEST_DIR, "Local test set")
    logger.info("Local test set: %d images, %d identities", ds.n_images, len(ds.identities))
    return ds


def make_synthetic_dataset(n_identities: int = 5, per_identity: int = 12,
                           dim: int = 128, seed: int = 42,
                           centre_scale: float = 0.048,
                           noise_std: float = 0.022) -> FaceDataset:
    """Fabricate well-separated embeddings and dummy images.

    Used for instant offline smoke tests of the whole evaluation pipeline
    without downloading data or running dlib. Embeddings are pre-filled so
    encode_dataset is skipped.

    The default centre_scale/noise_std are tuned so that, in 128-D, genuine
    pair distances cluster near ~0.35 and impostor distances near ~0.85 -
    the same scale produced by real face_recognition embeddings,
    so the offline charts look representative.
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, centre_scale, size=(n_identities, dim))
    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    ds = FaceDataset(source="Synthetic (offline)")
    for k in range(n_identities):
        for _ in range(per_identity):
            emb = centres[k] + rng.normal(0, noise_std, size=dim)
            embeddings.append(emb)
            labels.append(f"person_{k:02d}")
            # cheap, distinct grayscale pattern so LBPH has something to learn
            base = int(20 + k * (200 / max(1, n_identities)))
            gray = np.clip(
                np.full((160, 160), base, np.float32) + rng.normal(0, 8, (160, 160)),
                0, 255,
            ).astype(np.uint8)
            rgb, gray = _split_rgb_gray(gray)
            ds.images_rgb.append(rgb)
            ds.images_gray.append(gray)
    ds.labels = np.array(labels, dtype=object)
    ds.embeddings = np.array(embeddings, dtype=np.float64)
    return ds


def _merge(a: FaceDataset, b: FaceDataset) -> FaceDataset:
    merged = FaceDataset(source=f"{a.source} + {b.source}")
    merged.images_rgb = a.images_rgb + b.images_rgb
    merged.images_gray = a.images_gray + b.images_gray
    merged.labels = np.concatenate([a.labels, b.labels]) if a.n_images or b.n_images else a.labels
    return merged


def build_dataset(source: str = "auto") -> FaceDataset:
    """Assemble a FaceDataset according to the chosen source."""
    source = source.lower()
    if source == "synthetic":
        return make_synthetic_dataset()
    if source == "enrolled":
        return load_enrolled_dataset()
    if source == "local":
        return load_local_dataset()
    if source == "olivetti":
        return load_olivetti_dataset()
    if source == "lfw":
        try:
            return load_lfw_dataset()
        except Exception as exc:
            logger.warning("LFW load failed (%s); falling back to Olivetti", exc)
            return load_olivetti_dataset()

    # auto: a multi-identity base + the real enrolled subject if present
    local = load_local_dataset()
    if len(local.identities) >= 2:
        base = local
    else:
        try:
            base = load_lfw_dataset()
        except Exception as exc:
            logger.warning("LFW unavailable (%s); using Olivetti", exc)
            base = load_olivetti_dataset()

    enrolled = load_enrolled_dataset()
    if enrolled.n_images > 0:
        base = _merge(base, enrolled)
    return base


# Encoding

def encode_dataset(ds: FaceDataset) -> FaceDataset:
    """Compute 128-d dlib embeddings for every image (skips if pre-filled).

    Images that fail to encode are dropped from the dataset (and a warning
    logged) so all downstream arrays stay aligned.
    """
    if ds.embeddings is not None:
        return ds

    import face_recognition  # heavy import — deferred

    kept_rgb: list[np.ndarray] = []
    kept_gray: list[np.ndarray] = []
    kept_labels: list[str] = []
    embeddings: list[np.ndarray] = []

    for idx, (rgb, gray, label) in enumerate(
        zip(ds.images_rgb, ds.images_gray, ds.labels)
    ):
        emb = _encode_one(face_recognition, rgb)
        if emb is None:
            logger.warning("Encoding failed for image #%d (%s)", idx, label)
            continue
        embeddings.append(emb)
        kept_rgb.append(rgb)
        kept_gray.append(gray)
        kept_labels.append(label)

    ds.images_rgb = kept_rgb
    ds.images_gray = kept_gray
    ds.labels = np.array(kept_labels, dtype=object)
    ds.embeddings = np.array(embeddings, dtype=np.float64)
    logger.info("Encoded %d/%d images", ds.n_images, ds.n_images)
    return ds


def _encode_one(face_recognition, rgb: np.ndarray) -> np.ndarray | None:
    """Encode one image, detecting first and falling back to whole-frame."""
    try:
        locs = face_recognition.face_locations(rgb)
    except Exception:
        locs = []
    if locs:
        encs = face_recognition.face_encodings(rgb, known_face_locations=locs[:1])
        if encs:
            return encs[0]
    h, w = rgb.shape[:2]
    encs = face_recognition.face_encodings(rgb, known_face_locations=[(0, w, h, 0)])
    return encs[0] if encs else None


# Verification metrics: genuine/impostor, FAR/FRR, ROC, EER

def genuine_impostor_distances(
    embeddings: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Split all unordered image pairs into genuine and impostor distances.

    Returns:
        ``(genuine, impostor)`` 1-D arrays of euclidean distances, where a
        genuine pair shares the same identity label.
    """
    dist = pairwise_distances(embeddings, metric="euclidean")
    n = dist.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    pair_dist = dist[iu, ju]
    same = np.asarray(labels)[iu] == np.asarray(labels)[ju]
    return pair_dist[same], pair_dist[~same]


def threshold_sweep(
    genuine: np.ndarray, impostor: np.ndarray, thresholds: np.ndarray = DEFAULT_THRESHOLDS
) -> pd.DataFrame:
    """Compute FAR/FRR/TAR and pair-accuracy across thresholds.

    A pair is *accepted as the same person* when ``distance <= threshold``.

        FAR = impostor pairs accepted / total impostor pairs   (security)
        FRR = genuine pairs rejected / total genuine pairs      (convenience)
        TAR = 1 - FRR
    """
    genuine = np.asarray(genuine, dtype=float)
    impostor = np.asarray(impostor, dtype=float)
    rows = []
    n_gen = max(1, genuine.size)
    n_imp = max(1, impostor.size)
    total = genuine.size + impostor.size
    for t in thresholds:
        far = float(np.count_nonzero(impostor <= t)) / n_imp
        frr = float(np.count_nonzero(genuine > t)) / n_gen
        # accuracy over all pairs at this operating point
        correct = np.count_nonzero(genuine <= t) + np.count_nonzero(impostor > t)
        acc = correct / total if total else 0.0
        rows.append(
            {"threshold": float(t), "FAR": far, "FRR": frr, "TAR": 1.0 - frr,
             "pair_accuracy": acc}
        )
    return pd.DataFrame(rows)


def compute_eer(sweep: pd.DataFrame) -> tuple[float, float]:
    """Return ``(eer, eer_threshold)`` — the point where FAR ≈ FRR."""
    diff = (sweep["FAR"] - sweep["FRR"]).to_numpy()
    idx = int(np.argmin(np.abs(diff)))
    eer = float((sweep["FAR"].iloc[idx] + sweep["FRR"].iloc[idx]) / 2.0)
    return eer, float(sweep["threshold"].iloc[idx])


def optimal_threshold(sweep: pd.DataFrame) -> float:
    """Threshold that maximises pair-accuracy (a data-driven operating point)."""
    idx = int(sweep["pair_accuracy"].to_numpy().argmax())
    return float(sweep["threshold"].iloc[idx])


# Identification metrics (closed/open-set) with k-fold CV

def _safe_n_splits(labels: np.ndarray, requested: int) -> int:
    counts = pd.Series(labels).value_counts()
    return max(2, min(requested, int(counts.min())))


def identification_cv(
    embeddings: np.ndarray, labels: np.ndarray, threshold: float, n_splits: int = 5
) -> dict:
    """Nearest-neighbour identification with stratified k-fold CV.

    For each test embedding the nearest gallery (train) embedding is found;
    if that distance exceeds ``threshold`` the prediction is "unknown".
    Returns macro precision/recall/F1, accuracy, and pooled y_true/y_pred.
    """
    labels = np.asarray(labels, dtype=object)
    n_splits = _safe_n_splits(labels, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_true_all: list[str] = []
    y_pred_all: list[str] = []
    fold_acc: list[float] = []

    for train_idx, test_idx in skf.split(embeddings, labels):
        gallery = embeddings[train_idx]
        gallery_labels = labels[train_idx]
        d = pairwise_distances(embeddings[test_idx], gallery, metric="euclidean")
        nearest = d.argmin(axis=1)
        nearest_dist = d[np.arange(d.shape[0]), nearest]
        preds = np.where(
            nearest_dist <= threshold, gallery_labels[nearest], UNKNOWN_LABEL
        )
        truth = labels[test_idx]
        y_true_all.extend(truth.tolist())
        y_pred_all.extend(preds.tolist())
        fold_acc.append(accuracy_score(truth, preds))

    label_order = sorted(set(y_true_all) | set(y_pred_all))
    cm = confusion_matrix(y_true_all, y_pred_all, labels=label_order)
    return {
        "model": "dlib ResNet (face_recognition)",
        "threshold": threshold,
        "n_splits": n_splits,
        "accuracy": float(np.mean(fold_acc)),
        "accuracy_std": float(np.std(fold_acc)),
        "precision": float(precision_score(y_true_all, y_pred_all, average="macro",
                                           zero_division=0)),
        "recall": float(recall_score(y_true_all, y_pred_all, average="macro",
                                      zero_division=0)),
        "f1": float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)),
        "labels": label_order,
        "confusion_matrix": cm,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
    }


def lbph_cv(gray_images: list[np.ndarray], labels: np.ndarray, n_splits: int = 5) -> dict:
    """Closed-set identification accuracy for the OpenCV LBPH model via k-fold."""
    labels = np.asarray(labels, dtype=object)
    uniq = {name: i for i, name in enumerate(sorted(set(labels.tolist())))}
    inv = {i: name for name, i in uniq.items()}
    int_labels = np.array([uniq[name] for name in labels], dtype=np.int32)
    n_splits = _safe_n_splits(labels, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    prepared = [
        cv2.resize(g if g.ndim == 2 else cv2.cvtColor(g, cv2.COLOR_RGB2GRAY),
                   LBPH_FACE_SIZE, interpolation=cv2.INTER_AREA).astype(np.uint8)
        for g in gray_images
    ]

    y_true_all: list[str] = []
    y_pred_all: list[str] = []
    fold_acc: list[float] = []

    for train_idx, test_idx in skf.split(prepared, int_labels):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train([prepared[i] for i in train_idx], int_labels[train_idx])
        truth, preds = [], []
        for i in test_idx:
            pred_label, _conf = recognizer.predict(prepared[i])
            truth.append(inv[int(int_labels[i])])
            preds.append(inv.get(int(pred_label), UNKNOWN_LABEL))
        y_true_all.extend(truth)
        y_pred_all.extend(preds)
        fold_acc.append(accuracy_score(truth, preds))

    label_order = sorted(set(y_true_all) | set(y_pred_all))
    cm = confusion_matrix(y_true_all, y_pred_all, labels=label_order)
    return {
        "model": "OpenCV LBPH",
        "n_splits": n_splits,
        "accuracy": float(np.mean(fold_acc)),
        "accuracy_std": float(np.std(fold_acc)),
        "precision": float(precision_score(y_true_all, y_pred_all, average="macro",
                                           zero_division=0)),
        "recall": float(recall_score(y_true_all, y_pred_all, average="macro",
                                      zero_division=0)),
        "f1": float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)),
        "labels": label_order,
        "confusion_matrix": cm,
    }


# Plotting

def _ensure_out_dir() -> Path:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    return VALIDATION_DIR


def plot_distance_distributions(genuine, impostor, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(genuine, bins=30, color="#2ecc71", label="Genuine (same person)",
                 stat="density", kde=True, ax=ax)
    sns.histplot(impostor, bins=30, color="#e74c3c", label="Impostor (different)",
                 stat="density", kde=True, ax=ax)
    ax.axvline(config.RECOGNITION_THRESHOLD, color="black", linestyle="--",
               label=f"config threshold = {config.RECOGNITION_THRESHOLD}")
    ax.set_xlabel("Face embedding distance")
    ax.set_title("Genuine vs. Impostor distance distributions")
    ax.legend()
    fig.tight_layout()
    path = out / "distance_distributions.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_roc(sweep: pd.DataFrame, eer: float, eer_t: float, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(sweep["FAR"], sweep["TAR"], color="#3498db", label="ROC")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="chance")
    ax.scatter([eer], [1 - eer], color="red", zorder=5,
               label=f"EER = {eer:.3f} @ t={eer_t:.2f}")
    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("True Acceptance Rate (TAR)")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = out / "roc_curve.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_far_frr(sweep: pd.DataFrame, eer_t: float, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep["threshold"], sweep["FAR"], color="#e74c3c", label="FAR")
    ax.plot(sweep["threshold"], sweep["FRR"], color="#2ecc71", label="FRR")
    ax.axvline(eer_t, color="black", linestyle="--", label=f"EER threshold = {eer_t:.2f}")
    ax.axvline(config.RECOGNITION_THRESHOLD, color="purple", linestyle=":",
               label=f"config = {config.RECOGNITION_THRESHOLD}")
    ax.set_xlabel("Decision threshold (distance)")
    ax.set_ylabel("Error rate")
    ax.set_title("FAR / FRR vs. threshold")
    ax.legend()
    fig.tight_layout()
    path = out / "far_frr.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_confusion(result: dict, out: Path, fname: str) -> Path:
    cm = result["confusion_matrix"]
    labels = [l.replace(UNKNOWN_LABEL, "unknown") for l in result["labels"]]
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.7),
                                    max(4, len(labels) * 0.6)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {result['model']}")
    fig.tight_layout()
    path = out / fname
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_model_comparison(dlib_res: dict, lbph_res: dict, out: Path) -> Path:
    metrics = ["accuracy", "precision", "recall", "f1"]
    df = pd.DataFrame(
        {
            "metric": metrics * 2,
            "score": [dlib_res[m] for m in metrics] + [lbph_res[m] for m in metrics],
            "model": [dlib_res["model"]] * 4 + [lbph_res["model"]] * 4,
        }
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df, x="metric", y="score", hue="model", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("dlib ResNet vs. LBPH — identification metrics")
    fig.tight_layout()
    path = out / "model_comparison.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# Orchestration

def run_validation(source: str = "auto", n_splits: int = 5) -> dict:
    """Run the full evaluation and return a results dict; saves charts + JSON."""
    out = _ensure_out_dir()
    ds = build_dataset(source)

    if ds.n_images == 0:
        raise RuntimeError(
            "No data to validate. Enroll students, drop images in "
            f"{LOCAL_TEST_DIR}, or run with --source synthetic / olivetti."
        )

    ds = encode_dataset(ds)
    if len(ds.identities) < 2:
        raise RuntimeError(
            f"Need >= 2 identities for validation; got {ds.identities}. "
            "Add more people or use --source lfw / olivetti / synthetic."
        )

    logger.info("Validating on: %s (%d images, %d identities)",
                ds.source, ds.n_images, len(ds.identities))

    genuine, impostor = genuine_impostor_distances(ds.embeddings, ds.labels)
    sweep = threshold_sweep(genuine, impostor)
    eer, eer_t = compute_eer(sweep)
    opt_t = optimal_threshold(sweep)

    dlib_res = identification_cv(ds.embeddings, ds.labels,
                                 threshold=config.RECOGNITION_THRESHOLD, n_splits=n_splits)
    lbph_res = lbph_cv(ds.images_gray, ds.labels, n_splits=n_splits)

    far_at_cfg = float(np.count_nonzero(impostor <= config.RECOGNITION_THRESHOLD)
                       / max(1, impostor.size))
    frr_at_cfg = float(np.count_nonzero(genuine > config.RECOGNITION_THRESHOLD)
                       / max(1, genuine.size))

    charts = {
        "distance_distributions": str(plot_distance_distributions(genuine, impostor, out)),
        "roc_curve": str(plot_roc(sweep, eer, eer_t, out)),
        "far_frr": str(plot_far_frr(sweep, eer_t, out)),
        "confusion_dlib": str(plot_confusion(dlib_res, out, "confusion_dlib.png")),
        "confusion_lbph": str(plot_confusion(lbph_res, out, "confusion_lbph.png")),
        "model_comparison": str(plot_model_comparison(dlib_res, lbph_res, out)),
    }

    summary = {
        "data_source": ds.source,
        "n_images": ds.n_images,
        "n_identities": len(ds.identities),
        "config_threshold": config.RECOGNITION_THRESHOLD,
        "EER": round(eer, 4),
        "EER_threshold": round(eer_t, 4),
        "accuracy_optimal_threshold": round(opt_t, 4),
        "FAR_at_config_threshold": round(far_at_cfg, 4),
        "FRR_at_config_threshold": round(frr_at_cfg, 4),
        "genuine_mean_distance": round(float(np.mean(genuine)), 4),
        "impostor_mean_distance": round(float(np.mean(impostor)), 4),
        "dlib": {k: dlib_res[k] for k in
                 ("accuracy", "accuracy_std", "precision", "recall", "f1")},
        "lbph": {k: lbph_res[k] for k in
                 ("accuracy", "accuracy_std", "precision", "recall", "f1")},
        "charts": charts,
    }

    sweep.to_csv(out / "threshold_sweep.csv", index=False)
    with open(out / "metrics_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    _print_summary(summary)
    return {"summary": summary, "sweep": sweep, "dlib": dlib_res, "lbph": lbph_res,
            "genuine": genuine, "impostor": impostor, "dataset": ds}


def _print_summary(summary: dict) -> None:
    line = "=" * 64
    print("\n" + line)
    print("VisionGate — Validation Summary")
    print(line)
    print(f"  Data source            : {summary['data_source']}")
    print(f"  Images / identities    : {summary['n_images']} / {summary['n_identities']}")
    print("-" * 64)
    print(f"  Genuine mean distance  : {summary['genuine_mean_distance']}")
    print(f"  Impostor mean distance : {summary['impostor_mean_distance']}")
    print(f"  Equal Error Rate (EER) : {summary['EER']:.2%}  @ threshold {summary['EER_threshold']}")
    print(f"  Config threshold       : {summary['config_threshold']}")
    print(f"    -> FAR @ config      : {summary['FAR_at_config_threshold']:.2%}")
    print(f"    -> FRR @ config      : {summary['FRR_at_config_threshold']:.2%}")
    print(f"  Accuracy-optimal thr.  : {summary['accuracy_optimal_threshold']}")
    print("-" * 64)
    d, l = summary["dlib"], summary["lbph"]
    print(f"  {'Model':<26}{'Acc':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}")
    print(f"  {'dlib ResNet':<26}{d['accuracy']:>8.3f}{d['precision']:>8.3f}"
          f"{d['recall']:>8.3f}{d['f1']:>8.3f}")
    print(f"  {'OpenCV LBPH':<26}{l['accuracy']:>8.3f}{l['precision']:>8.3f}"
          f"{l['recall']:>8.3f}{l['f1']:>8.3f}")
    print("-" * 64)
    print(f"  Charts + JSON saved to : {VALIDATION_DIR}")
    print(line + "\n")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VisionGate validation suite")
    parser.add_argument(
        "--source", default="auto",
        choices=["auto", "lfw", "olivetti", "enrolled", "local", "synthetic"],
        help="Which dataset to validate against (default: auto)",
    )
    parser.add_argument("--folds", type=int, default=5, help="k for k-fold CV")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_validation(source=args.source, n_splits=args.folds)


if __name__ == "__main__":
    main()
