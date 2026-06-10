"""Robustness analysis for the recognition pipeline.

Measures how recognition holds up under controlled perturbations of the
real LFW test images, plus a couple of extra evaluation curves. All of it
reuses the same encoder/matching path as validate.py.

Produces in data/validation/:
    roc_auc.png              ROC with the AUC score
    learning_curve.png       accuracy vs. gallery size per person
    robustness_lighting.png  accuracy under darker/brighter conditions
    robustness_distance.png  accuracy as the face gets lower-res (farther)
    robustness_occlusion.png accuracy as more of the face is covered
    robustness_expressions.png  (only if you captured expression photos)
    robustness_summary.json

Usage:
    python validate_robustness.py                  # run the analysis
    python validate_robustness.py --capture-expressions   # take expression photos first
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances

import config
from utils import get_logger
from validate import (
    UNKNOWN_LABEL,
    VALIDATION_DIR,
    _encode_one,
    build_dataset,
    encode_dataset,
    genuine_impostor_distances,
)

logger = get_logger(__name__)

EXPRESSIONS_DIR = config.PROJECT_ROOT / "data" / "expressions"
THRESHOLD = config.RECOGNITION_THRESHOLD


# --- perturbations -------------------------------------------------------

def perturb_brightness(img: np.ndarray, gain: float) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=gain, beta=0)


def perturb_resolution(img: np.ndarray, scale: float) -> np.ndarray:
    """Downscale then upscale back — simulates a smaller/farther face."""
    if scale >= 1.0:
        return img
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def perturb_occlusion(img: np.ndarray, frac: float) -> np.ndarray:
    """Black out the top ``frac`` of the face (covers brow/eyes as it grows)."""
    if frac <= 0:
        return img
    out = img.copy()
    h = out.shape[0]
    out[: int(h * frac), :] = 0
    return out


# --- encoding helpers ----------------------------------------------------

def _encode_list(images_rgb: list[np.ndarray]) -> list[np.ndarray | None]:
    import face_recognition
    return [_encode_one(face_recognition, im) for im in images_rgb]


def _identify(probe_emb: list[np.ndarray | None], probe_labels, gallery_emb,
              gallery_labels, threshold: float = THRESHOLD) -> float:
    """1-NN identification accuracy; a failed/too-far encoding = wrong."""
    preds = []
    for emb in probe_emb:
        if emb is None:
            preds.append(UNKNOWN_LABEL)
            continue
        d = pairwise_distances([emb], gallery_emb)[0]
        j = int(d.argmin())
        preds.append(gallery_labels[j] if d[j] <= threshold else UNKNOWN_LABEL)
    return accuracy_score(list(probe_labels), preds)


# --- analyses ------------------------------------------------------------

def run_condition(name: str, levels, label_fmt, perturb_fn, probe_imgs,
                  probe_labels, gallery_emb, gallery_labels) -> list[tuple[str, float]]:
    results = []
    for lvl in levels:
        perturbed = [perturb_fn(im, lvl) for im in probe_imgs]
        emb = _encode_list(perturbed)
        acc = _identify(emb, probe_labels, gallery_emb, gallery_labels)
        results.append((label_fmt(lvl), acc))
        logger.info("%s = %s -> accuracy %.3f", name, label_fmt(lvl), acc)
    return results


def learning_curve(embeddings, labels) -> list[tuple[int, float]]:
    """Accuracy vs. number of gallery images per identity (honest stand-in
    for a training-loss curve, since the encoder is pre-trained)."""
    labels = np.asarray(labels, dtype=object)
    by_id: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        by_id.setdefault(lab, []).append(i)
    min_count = min(len(v) for v in by_id.values())

    out = []
    for k in range(1, min_count):
        g_idx, p_idx = [], []
        for idxs in by_id.values():
            g_idx += idxs[:k]
            p_idx += idxs[k:]
        acc = _identify([embeddings[i] for i in p_idx], labels[p_idx],
                        embeddings[g_idx], labels[g_idx])
        out.append((k, acc))
    return out


def expression_test() -> list[tuple[str, float, bool]]:
    """Distance from each captured expression to a neutral reference."""
    if not EXPRESSIONS_DIR.exists():
        return []
    files = sorted(p for p in EXPRESSIONS_DIR.iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if len(files) < 2:
        logger.warning("Need >= 2 expression photos in %s", EXPRESSIONS_DIR)
        return []

    import face_recognition
    imgs = {p.stem: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in files}
    encs = {k: _encode_one(face_recognition, v) for k, v in imgs.items()}
    encs = {k: v for k, v in encs.items() if v is not None}
    if "neutral" in encs:
        ref_name = "neutral"
    else:
        ref_name = next(iter(encs))
    ref = encs[ref_name]

    out = []
    for name, e in encs.items():
        if name == ref_name:
            continue
        dist = float(np.linalg.norm(e - ref))
        out.append((name, dist, dist <= THRESHOLD))
        logger.info("expression %s: dist=%.3f to %s", name, dist, ref_name)
    return out


# --- plotting ------------------------------------------------------------

def _bar(results, title, xlabel, path: Path) -> Path:
    labels = [r[0] for r in results]
    accs = [r[1] for r in results]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, accs, color="#38bdf8")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.2f}",
                ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Identification accuracy")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_roc_auc(genuine, impostor, path: Path) -> float:
    y_true = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
    y_score = np.r_[-np.asarray(genuine), -np.asarray(impostor)]  # closer = higher
    auc = float(roc_auc_score(y_true, y_score))
    fpr, tpr, _ = roc_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color="#3498db", label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="chance")
    ax.set_xlabel("False Acceptance Rate")
    ax.set_ylabel("True Acceptance Rate")
    ax.set_title("ROC curve with AUC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return auc


def plot_learning_curve(points, path: Path) -> Path:
    ks = [p[0] for p in points]
    accs = [p[1] for p in points]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, accs, marker="o", color="#a78bfa")
    ax.set_xlabel("Gallery images per person")
    ax.set_ylabel("Identification accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Learning curve — accuracy vs. enrolled images per person")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_expressions(results, path: Path) -> Path:
    names = [r[0] for r in results]
    dists = [r[1] for r in results]
    colors = ["#16a34a" if r[2] else "#e74c3c" for r in results]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, dists, color=colors)
    ax.axhline(THRESHOLD, color="black", linestyle="--",
               label=f"threshold = {THRESHOLD}")
    ax.set_ylabel("Distance to neutral reference")
    ax.set_xlabel("Expression")
    ax.set_title("Recognition distance per expression (green = recognized)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# --- expression capture --------------------------------------------------

def capture_expressions() -> None:
    names = ["neutral", "smile", "surprise", "mouth_open", "eyes_squint"]
    EXPRESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera at index {config.CAMERA_INDEX}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    win = "Capture expressions"
    print("For each prompt: make the expression, press SPACE to save, S to skip, ESC to quit.")
    for name in names:
        saved = False
        while not saved:
            ok, frame = cap.read()
            if not ok:
                continue
            preview = cv2.flip(frame, 1)
            for txt, y in ((f"Expression: {name}", 30),
                           ("SPACE=save  S=skip  ESC=quit", 60)):
                cv2.putText(preview, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(preview, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:       # SPACE
                cv2.imwrite(str(EXPRESSIONS_DIR / f"{name}.png"), frame)
                print(f"  saved {name}.png")
                saved = True
            elif key in (ord("s"), ord("S")):
                print(f"  skipped {name}")
                break
            elif key == 27:     # ESC
                cap.release()
                cv2.destroyAllWindows()
                print("Capture stopped.")
                return
    cap.release()
    cv2.destroyAllWindows()
    print(f"Expression photos saved in {EXPRESSIONS_DIR}")


# --- orchestration -------------------------------------------------------

def run() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    ds = encode_dataset(build_dataset("lfw"))
    if len(ds.identities) < 2:
        raise SystemExit("Need >= 2 identities; LFW load failed?")

    logger.info("Robustness base: %d images, %d identities",
                ds.n_images, len(ds.identities))

    emb = ds.embeddings
    labels = np.asarray(ds.labels, dtype=object)
    idx = np.arange(len(labels))
    g_idx, p_idx = train_test_split(idx, test_size=0.5, stratify=labels,
                                    random_state=42)
    gallery_emb, gallery_labels = emb[g_idx], labels[g_idx]
    probe_imgs = [ds.images_rgb[i] for i in p_idx]
    probe_labels = labels[p_idx]

    summary: dict = {"data_source": ds.source, "n_images": ds.n_images,
                     "n_identities": len(ds.identities), "threshold": THRESHOLD}

    # AUC
    genuine, impostor = genuine_impostor_distances(emb, labels)
    summary["AUC"] = round(plot_roc_auc(genuine, impostor,
                                        VALIDATION_DIR / "roc_auc.png"), 4)

    # Learning curve
    lc = learning_curve(emb, labels)
    plot_learning_curve(lc, VALIDATION_DIR / "learning_curve.png")
    summary["learning_curve"] = [{"gallery_per_person": k, "accuracy": round(a, 4)}
                                 for k, a in lc]

    # Lighting
    light = run_condition(
        "lighting", [0.4, 0.7, 1.0, 1.3, 1.6],
        lambda g: f"x{g:g}", perturb_brightness,
        probe_imgs, probe_labels, gallery_emb, gallery_labels)
    _bar(light, "Accuracy under different lighting (brightness gain)",
         "Brightness gain", VALIDATION_DIR / "robustness_lighting.png")
    summary["lighting"] = [{"level": l, "accuracy": round(a, 4)} for l, a in light]

    # Distance (resolution proxy)
    dist = run_condition(
        "distance", [1.0, 0.6, 0.4, 0.25],
        lambda s: f"{int(s*100)}%", perturb_resolution,
        probe_imgs, probe_labels, gallery_emb, gallery_labels)
    _bar(dist, "Accuracy vs. distance (face resolution)",
         "Face resolution (lower = farther)", VALIDATION_DIR / "robustness_distance.png")
    summary["distance"] = [{"level": l, "accuracy": round(a, 4)} for l, a in dist]

    # Occlusion
    occ = run_condition(
        "occlusion", [0.0, 0.2, 0.4, 0.6],
        lambda f: f"{int(f*100)}%", perturb_occlusion,
        probe_imgs, probe_labels, gallery_emb, gallery_labels)
    _bar(occ, "Accuracy under partial face occlusion",
         "Fraction of face covered", VALIDATION_DIR / "robustness_occlusion.png")
    summary["occlusion"] = [{"level": l, "accuracy": round(a, 4)} for l, a in occ]

    # Expressions (only if captured)
    expr = expression_test()
    if expr:
        plot_expressions(expr, VALIDATION_DIR / "robustness_expressions.png")
        summary["expressions"] = [{"expression": n, "distance": round(d, 4),
                                   "recognized": r} for n, d, r in expr]
    else:
        summary["expressions"] = "no expression photos captured"

    with open(VALIDATION_DIR / "robustness_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    _print_summary(summary)


def _print_summary(s: dict) -> None:
    line = "=" * 60
    print("\n" + line)
    print("Robustness Summary")
    print(line)
    print(f"  Data            : {s['data_source']} ({s['n_images']} imgs, "
          f"{s['n_identities']} ids)")
    print(f"  AUC             : {s['AUC']}")
    print("  Lighting        : " + ", ".join(f"{d['level']}={d['accuracy']:.2f}"
                                             for d in s["lighting"]))
    print("  Distance        : " + ", ".join(f"{d['level']}={d['accuracy']:.2f}"
                                             for d in s["distance"]))
    print("  Occlusion       : " + ", ".join(f"{d['level']}={d['accuracy']:.2f}"
                                             for d in s["occlusion"]))
    if isinstance(s["expressions"], list):
        print("  Expressions     : " + ", ".join(
            f"{e['expression']}={'OK' if e['recognized'] else 'X'}"
            for e in s["expressions"]))
    else:
        print(f"  Expressions     : {s['expressions']}")
    print(f"  Charts + JSON   : {VALIDATION_DIR}")
    print(line + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="VisionGate robustness analysis")
    parser.add_argument("--capture-expressions", action="store_true",
                        help="Capture expression photos from the webcam first")
    args = parser.parse_args(argv)
    if args.capture_expressions:
        capture_expressions()
        return
    run()


if __name__ == "__main__":
    main()
