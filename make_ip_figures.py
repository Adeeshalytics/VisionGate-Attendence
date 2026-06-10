"""Generate the image-processing analysis figures for the report/slides.

Produces four PNGs in data/validation/ from a real face image, using the
same functions the pipeline uses (preprocess.py / anti_spoof.py):

    ip_clahe.png         original vs CLAHE-enhanced face
    ip_histogram.png     intensity histogram before/after CLAHE
    ip_augmentation.png  the 7 augmentation variants
    ip_ear_blink.png     real eye landmarks + EAR, and a blink-detection trace

Pick the source face one of three ways:

    python make_ip_figures.py --capture          # take a fresh webcam photo
    python make_ip_figures.py --image me.jpg     # use an existing photo
    python make_ip_figures.py                     # fall back to an enrolled crop

The face is auto-detected and cropped, so background/clothing don't show.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import preprocess
from utils import get_logger

logger = get_logger(__name__)

OUT = config.PROJECT_ROOT / "data" / "validation"
FACE_SIZE = (240, 240)


def _crop_to_face(frame: np.ndarray) -> np.ndarray:
    """Detect the largest face and return a padded square crop. Falls back to
    a centre crop of the whole frame if no face is found."""
    from enroll import (
        build_haar_classifier, build_mediapipe_detector, crop_face_roi, detect_faces,
    )

    mp_detector = build_mediapipe_detector()
    haar = build_haar_classifier()
    try:
        boxes = detect_faces(frame, mp_detector, haar)
    finally:
        try:
            mp_detector.close()
        except Exception:
            pass

    if boxes:
        # largest box by area; pad generously so head + shoulders (and the
        # t-shirt) are in frame, matching the portrait framing we want.
        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        pad = int(0.55 * max(w, h))
        crop = crop_face_roi(frame, (x, y, w, h), padding=pad)
    else:
        logger.warning("No face detected; using a centre crop of the frame.")
        h, w = frame.shape[:2]
        s = min(h, w)
        crop = frame[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]

    return cv2.resize(crop, FACE_SIZE)


def _detect_face_bbox(frame: np.ndarray):
    """Return the largest (x, y, w, h) face box in ``frame``, or None."""
    from enroll import build_haar_classifier, build_mediapipe_detector, detect_faces

    mp_detector = build_mediapipe_detector()
    haar = build_haar_classifier()
    try:
        boxes = detect_faces(frame, mp_detector, haar)
    finally:
        try:
            mp_detector.close()
        except Exception:
            pass
    if not boxes:
        return None
    return max(boxes, key=lambda b: b[2] * b[3])


def _capture_from_webcam() -> np.ndarray:
    """Open the webcam, let the user frame up, SPACE to grab, ESC to cancel."""
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera at index {config.CAMERA_INDEX}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    win = "Capture face  —  SPACE = take photo,  ESC = cancel"
    grabbed = None
    print("Look at the camera in good light, then press SPACE. ESC to cancel.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        preview = cv2.flip(frame, 1)  # mirror for a natural preview
        cv2.putText(preview, "SPACE = capture   ESC = cancel", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(preview, "SPACE = capture   ESC = cancel", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, preview)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:      # SPACE
            grabbed = frame.copy()  # keep the un-mirrored original
            break
        if key == 27:      # ESC
            break
    cap.release()
    cv2.destroyAllWindows()
    if grabbed is None:
        raise SystemExit("Capture cancelled.")
    return grabbed


def _load_source(image: str | None, capture: bool) -> np.ndarray:
    """Resolve the source face from --capture, --image, or an enrolled crop."""
    if capture:
        return _crop_to_face(_capture_from_webcam())

    if image:
        path = Path(image)
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise SystemExit(f"Could not read image: {path}")
        logger.info("Using source image %s", path)
        return _crop_to_face(bgr)

    for person in sorted(config.ENROLLED_FACES_DIR.glob("*")):
        if not person.is_dir():
            continue
        for img in sorted(person.glob("*.png")):
            bgr = cv2.imread(str(img))
            if bgr is not None:
                logger.info("Using enrolled crop %s", img)
                return cv2.resize(bgr, FACE_SIZE)
    raise SystemExit(
        "No source face. Use --capture, --image <path>, or enroll someone first."
    )


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def figure_clahe(face_bgr: np.ndarray) -> Path:
    gray = preprocess.to_grayscale(face_bgr)
    enhanced = preprocess.apply_clahe(gray)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes[0].imshow(_bgr_to_rgb(face_bgr))
    axes[0].set_title("Original (BGR)")
    axes[1].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Grayscale")
    axes[2].imshow(enhanced, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("CLAHE (clip=2.0, 8x8 tiles)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Contrast enhancement with CLAHE", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = OUT / "ip_clahe.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_histogram(face_bgr: np.ndarray) -> Path:
    gray = preprocess.to_grayscale(face_bgr)
    enhanced = preprocess.apply_clahe(gray)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(gray.ravel(), bins=64, range=(0, 255), alpha=0.55,
            color="#64748b", label="Before CLAHE")
    ax.hist(enhanced.ravel(), bins=64, range=(0, 255), alpha=0.55,
            color="#38bdf8", label="After CLAHE")
    ax.set_xlabel("Pixel intensity (0–255)")
    ax.set_ylabel("Pixel count")
    ax.set_title("Intensity histogram — CLAHE spreads the dynamic range")
    ax.legend()
    fig.tight_layout()
    path = OUT / "ip_histogram.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_augmentation(face_bgr: np.ndarray) -> Path:
    variants = preprocess.augment_image(face_bgr)
    titles = [
        "original", "h-flip", "rotate +15", "rotate -15",
        "bright +40", "bright -40", "gaussian noise",
    ]
    n = len(variants)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3 * rows))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(_bgr_to_rgb(variants[i]))
            ax.set_title(titles[i] if i < len(titles) else f"aug {i}", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Data augmentation — {n} variants per captured face",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = OUT / "ip_augmentation.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _synthetic_ear_trace(n_frames: int = 80, seed: int = 7) -> np.ndarray:
    """An illustrative EAR-vs-frame signal: open-eye baseline with two blinks."""
    rng = np.random.default_rng(seed)
    ear = 0.31 + rng.normal(0, 0.008, n_frames)
    for centre in (22, 55):  # two blinks
        for k, val in zip(range(centre - 1, centre + 3), (0.22, 0.11, 0.12, 0.24)):
            if 0 <= k < n_frames:
                ear[k] = val
    return ear


def _count_blinks(ear: np.ndarray) -> list[int]:
    """Replicate the anti_spoof blink rule: a run of frames below threshold of
    at least EAR_CONSEC_FRAMES counts as one blink. Returns the frame index
    where each blink is registered (when the eye reopens)."""
    below = 0
    detected: list[int] = []
    for i, e in enumerate(ear):
        if e < config.EAR_THRESHOLD:
            below += 1
        else:
            if below >= config.EAR_CONSEC_FRAMES:
                detected.append(i)
            below = 0
    return detected


def figure_ear_blink(face_bgr: np.ndarray) -> Path:
    fig, (ax_face, ax_plot) = plt.subplots(
        1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 2]}
    )

    # Left: real eye landmarks + measured EAR on the actual face.
    annotated = face_bgr.copy()
    measured_ear = None
    try:
        import dlib
        from anti_spoof import (
            LEFT_EYE_IDX, RIGHT_EYE_IDX, _ensure_landmark_model, get_ear,
        )

        predictor = dlib.shape_predictor(str(_ensure_landmark_model()))
        gray = preprocess.to_grayscale(face_bgr)
        h, w = gray.shape[:2]
        # Locate the face within the (head-and-shoulders) crop so the
        # landmarks land on the eyes, not the whole frame.
        bbox = _detect_face_bbox(face_bgr)
        if bbox is not None:
            bx, by, bw, bh = bbox
            rect = dlib.rectangle(bx, by, bx + bw, by + bh)
        else:
            rect = dlib.rectangle(0, 0, w, h)
        shape = predictor(gray, rect)
        pts = np.array([(shape.part(i).x, shape.part(i).y)
                        for i in range(shape.num_parts)])
        for idx in list(LEFT_EYE_IDX) + list(RIGHT_EYE_IDX):
            cv2.circle(annotated, tuple(pts[idx]), 2, (0, 255, 0), -1)
        measured_ear = (get_ear(pts[list(LEFT_EYE_IDX)])
                        + get_ear(pts[list(RIGHT_EYE_IDX)])) / 2.0
    except Exception as exc:
        logger.warning("Could not draw eye landmarks: %s", exc)

    ax_face.imshow(_bgr_to_rgb(annotated))
    ax_face.axis("off")
    title = "Eye landmarks (dlib 68-pt)"
    if measured_ear is not None:
        title += f"\nmeasured EAR = {measured_ear:.3f}"
    ax_face.set_title(title, fontsize=11)

    # Right: EAR trace with threshold and detected blinks.
    ear = _synthetic_ear_trace()
    blinks = _count_blinks(ear)
    frames = np.arange(len(ear))
    ax_plot.plot(frames, ear, color="#3498db", label="EAR")
    ax_plot.axhline(config.EAR_THRESHOLD, color="#e74c3c", linestyle="--",
                    label=f"threshold = {config.EAR_THRESHOLD}")
    for b in blinks:
        ax_plot.annotate("blink", xy=(b, ear[b]), xytext=(b, ear[b] + 0.06),
                         ha="center", color="#16a34a", fontsize=9,
                         arrowprops=dict(arrowstyle="->", color="#16a34a"))
    ax_plot.set_xlabel("Frame")
    ax_plot.set_ylabel("Eye Aspect Ratio")
    ax_plot.set_ylim(0, 0.4)
    ax_plot.set_title(f"Blink detection — {len(blinks)} blink(s) "
                      f"(>= {config.EAR_CONSEC_FRAMES} frames below threshold)")
    ax_plot.legend(loc="lower right")

    fig.suptitle("Liveness: Eye Aspect Ratio (EAR) blink detection",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = OUT / "ip_ear_blink.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate image-processing figures")
    parser.add_argument("--capture", action="store_true",
                        help="Take a fresh photo from the webcam")
    parser.add_argument("--image", help="Use an existing photo file instead")
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    face = _load_source(args.image, args.capture)

    # Save the chosen source face so you can see exactly what was used.
    cv2.imwrite(str(OUT / "ip_source_face.png"), face)

    made = [
        figure_clahe(face),
        figure_histogram(face),
        figure_augmentation(face),
        figure_ear_blink(face),
    ]
    print("\nImage-processing figures written to", OUT)
    for p in made:
        print("  -", p.name)


if __name__ == "__main__":
    main()
