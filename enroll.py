"""Enrollment: capture a student's face, build the recognition gallery.

Grabs frames from the webcam, computes 128-d face_recognition (dlib) embeddings
and also trains an OpenCV LBPH model as a second opinion. Detection uses
MediaPipe (BlazeFace short-range) first and falls back to OpenCV's Haar cascade
when MediaPipe finds nothing in a frame.

    python enroll.py
"""

from __future__ import annotations

import pickle
import re
import urllib.request
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import cv2
import numpy as np

import config
import database
import preprocess
from utils import draw_bounding_box, ensure_dirs, get_logger

logger = get_logger(__name__)

# MediaPipe Tasks API (the legacy `mp.solutions` namespace was dropped in
# 0.10.35 wheels for Python 3.14, so we use the modern API).
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# BlazeFace short-range model for the Tasks API. The file is fetched on
# first use and cached under ``models/``.
_BLAZE_FACE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
_BLAZE_FACE_PATH: Path = config.PROJECT_ROOT / "models" / "blaze_face_short_range.tflite"

INSTRUCTIONS: List[str] = [
    "Look straight",
    "Turn slightly left",
    "Turn slightly right",
    "Look up slightly",
]
INSTRUCTION_INTERVAL = 30  # frames captured between instruction changes

LBPH_FACE_SIZE: Tuple[int, int] = (200, 200)
DETECTION_BOX_COLOR_BGR = (255, 128, 0)  # BLUE-ish (BGR)


# Detection

def _ensure_blaze_face_model() -> Path:
    """Download the BlazeFace short-range tflite model if missing.

    Returns the local path on disk.
    """
    _BLAZE_FACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _BLAZE_FACE_PATH.exists():
        logger.info("Downloading BlazeFace model to %s", _BLAZE_FACE_PATH)
        urllib.request.urlretrieve(_BLAZE_FACE_URL, _BLAZE_FACE_PATH)
    return _BLAZE_FACE_PATH


def build_mediapipe_detector() -> mp_vision.FaceDetector:
    """Construct a MediaPipe ``FaceDetector`` with the configured confidence."""
    model_path = _ensure_blaze_face_model()
    options = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        min_detection_confidence=config.MIN_FACE_CONFIDENCE,
    )
    return mp_vision.FaceDetector.create_from_options(options)


def build_haar_classifier() -> cv2.CascadeClassifier:
    """Load OpenCV's bundled Haar frontal-face cascade."""
    classifier = cv2.CascadeClassifier(str(config.HAAR_CASCADE_PATH))
    if classifier.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {config.HAAR_CASCADE_PATH}")
    return classifier


def detect_face_mediapipe(
    frame: np.ndarray, detector: mp_vision.FaceDetector
) -> List[Tuple[int, int, int, int]]:
    """Detect faces in ``frame`` using MediaPipe.

    Args:
        frame: BGR image.
        detector: a FaceDetector built by build_mediapipe_detector.

    Returns:
        A list of ``(x, y, w, h)`` boxes in absolute pixel coordinates,
        clamped to the frame.
    """
    if frame is None or detector is None:
        return []

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    h, w = frame.shape[:2]
    boxes: List[Tuple[int, int, int, int]] = []
    for detection in getattr(result, "detections", []) or []:
        bbox = detection.bounding_box
        x = max(0, int(bbox.origin_x))
        y = max(0, int(bbox.origin_y))
        bw = max(0, int(bbox.width))
        bh = max(0, int(bbox.height))
        bw = min(bw, w - x)
        bh = min(bh, h - y)
        if bw >= config.MIN_FACE_SIZE[0] and bh >= config.MIN_FACE_SIZE[1]:
            boxes.append((x, y, bw, bh))
    return boxes


def detect_face_haar(
    frame: np.ndarray, classifier: cv2.CascadeClassifier
) -> List[Tuple[int, int, int, int]]:
    """Fallback Haar Cascade detector.

    Args:
        frame: BGR image (grayscale conversion is handled internally).
        classifier: A loaded ``cv2.CascadeClassifier``.

    Returns:
        A list of ``(x, y, w, h)`` boxes.
    """
    if frame is None or classifier is None:
        return []
    gray = preprocess.to_grayscale(frame)
    detections = classifier.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=config.MIN_FACE_SIZE,
    )
    return [tuple(int(v) for v in box) for box in detections]


def detect_faces(
    frame: np.ndarray,
    mp_detector: mp_vision.FaceDetector,
    haar_classifier: cv2.CascadeClassifier,
) -> List[Tuple[int, int, int, int]]:
    """Run MediaPipe first; fall back to Haar if nothing is found.

    Logs which detector produced the boxes.
    """
    boxes = detect_face_mediapipe(frame, mp_detector)
    if boxes:
        logger.debug("MediaPipe detected %d face(s)", len(boxes))
        return boxes

    boxes = detect_face_haar(frame, haar_classifier)
    if boxes:
        logger.debug("Haar fallback detected %d face(s)", len(boxes))
    return boxes


# Cropping & encodings

def crop_face_roi(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: int = 20,
) -> np.ndarray:
    """Crop the face ROI from ``frame`` with ``padding`` pixels around the box.

    The padded rectangle is clamped to the frame so the result is always
    inside ``frame``.
    """
    if frame is None:
        raise ValueError("frame is None")
    if len(bbox) != 4:
        raise ValueError("bbox must have 4 values (x, y, w, h)")

    x, y, w, h = (int(v) for v in bbox)
    fh, fw = frame.shape[:2]

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(fw, x + w + padding)
    y2 = min(fh, y + h + padding)

    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, frame.shape[2]) if frame.ndim == 3 else (1, 1), dtype=frame.dtype)

    return frame[y1:y2, x1:x2].copy()


def compute_encodings(image_list: Iterable[np.ndarray]) -> Tuple[List[np.ndarray], int]:
    """Compute 128-d face_recognition encodings for each image.

    Each input is treated as an already-cropped face: we pass an explicit
    ``known_face_locations`` covering the whole frame so that
    face_recognition skips its HOG re-detection. This dramatically
    reduces dropouts on augmented variants (rotation, brightness, noise)
    where the HOG detector would otherwise fail to re-find the face.

    Returns:
        ``(encodings, failures)`` where ``encodings`` only contains the
        successful results.
    """
    import face_recognition  # heavy import — deferred to call time

    encodings: List[np.ndarray] = []
    failures = 0
    for idx, image in enumerate(image_list):
        if image is None or image.size == 0:
            failures += 1
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        h, w = rgb.shape[:2]
        locations = [(0, w, h, 0)]  # (top, right, bottom, left)
        try:
            results = face_recognition.face_encodings(
                rgb, known_face_locations=locations
            )
        except Exception as exc:
            failures += 1
            logger.warning("Encoding raised on image #%d: %s", idx, exc)
            continue
        if not results:
            failures += 1
            logger.warning("Encoding failed for image #%d", idx)
            continue
        encodings.append(results[0])
    return encodings, failures


def save_encodings(
    encodings: List[np.ndarray],
    labels: List[str],
    path: Path,
    names: dict[str, str] | None = None,
) -> None:
    """Append ``encodings``/``labels`` into the pickle at ``path``.

    The pickle holds::

        {
            "encodings": list[np.ndarray],   # 128-d vectors
            "labels":    list[str],           # student_id per encoding
            "names":     dict[str, str],      # student_id -> display name
        }

    Existing data is loaded and merged. The file's parent directory is
    created if missing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {"encodings": [], "labels": [], "names": {}}
    if path.exists():
        try:
            with open(path, "rb") as fh:
                loaded = pickle.load(fh)
            if isinstance(loaded, dict):
                data["encodings"] = list(loaded.get("encodings", []))
                data["labels"] = list(loaded.get("labels", []))
                data["names"] = dict(loaded.get("names", {}))
        except (pickle.UnpicklingError, EOFError) as exc:
            logger.warning("Could not read existing encodings at %s: %s", path, exc)

    data["encodings"].extend(encodings)
    data["labels"].extend(labels)
    if names:
        data["names"].update(names)

    with open(path, "wb") as fh:
        pickle.dump(data, fh)
    logger.info(
        "Saved encodings to %s (total=%d, students=%d)",
        path,
        len(data["encodings"]),
        len(data["names"]),
    )


def train_lbph(
    gray_face_images: List[np.ndarray],
    int_labels: List[int],
    path: Path,
) -> None:
    """Train an OpenCV LBPH recognizer and save it to path.

    Images are resized to LBPH_FACE_SIZE. If a model already exists at path it
    gets updated with the new samples; otherwise we train a fresh one.
    """
    if len(gray_face_images) == 0:
        logger.warning("train_lbph called with no images; skipping")
        return
    if len(gray_face_images) != len(int_labels):
        raise ValueError("gray_face_images and int_labels must be the same length")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    prepared: List[np.ndarray] = []
    for img in gray_face_images:
        if img is None or img.size == 0:
            continue
        if img.ndim == 3:
            img = preprocess.to_grayscale(img)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        prepared.append(cv2.resize(img, LBPH_FACE_SIZE, interpolation=cv2.INTER_AREA))

    labels_arr = np.array(int_labels[: len(prepared)], dtype=np.int32)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if path.exists():
        try:
            recognizer.read(str(path))
            recognizer.update(prepared, labels_arr)
            logger.info("Updated existing LBPH model with %d samples", len(prepared))
        except cv2.error as exc:
            logger.warning("Could not update existing LBPH model (%s); retraining", exc)
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(prepared, labels_arr)
    else:
        recognizer.train(prepared, labels_arr)
        logger.info("Trained new LBPH model with %d samples", len(prepared))

    recognizer.save(str(path))
    logger.info("Saved LBPH model to %s", path)


# LBPH label bookkeeping

_LBPH_LABEL_MAP_PATH: Path = config.PROJECT_ROOT / "data" / "encodings" / "lbph_labels.pkl"


def _load_lbph_label_map() -> dict[str, int]:
    if _LBPH_LABEL_MAP_PATH.exists():
        try:
            with open(_LBPH_LABEL_MAP_PATH, "rb") as fh:
                data = pickle.load(fh)
            if isinstance(data, dict):
                return {str(k): int(v) for k, v in data.items()}
        except (pickle.UnpicklingError, EOFError) as exc:
            logger.warning("Could not read LBPH label map: %s", exc)
    return {}


def _save_lbph_label_map(mapping: dict[str, int]) -> None:
    _LBPH_LABEL_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_LBPH_LABEL_MAP_PATH, "wb") as fh:
        pickle.dump(mapping, fh)


def _next_lbph_label(mapping: dict[str, int]) -> int:
    return (max(mapping.values()) + 1) if mapping else 0


# Persistence: raw face images on disk

def _save_raw_faces(student_id: str, raw_crops: List[np.ndarray]) -> Path:
    student_dir = config.ENROLLED_FACES_DIR / student_id
    student_dir.mkdir(parents=True, exist_ok=True)
    for idx, crop in enumerate(raw_crops):
        if crop is None or crop.size == 0:
            continue
        cv2.imwrite(str(student_dir / f"{student_id}_{idx:03d}.png"), crop)
    return student_dir


# Display helpers

def _overlay_text(frame: np.ndarray, text: str, origin: Tuple[int, int]) -> None:
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _flash(frame: np.ndarray) -> np.ndarray:
    """Return a white-blended copy of ``frame`` for capture feedback."""
    white = np.full_like(frame, 255)
    return cv2.addWeighted(frame, 0.3, white, 0.7, 0)


# Orchestrator

_STUDENT_ID_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _sanitize_student_id(raw: str) -> str:
    """Normalise a student ID into a filesystem-safe slug.

    Characters outside ``[A-Za-z0-9_-]`` are collapsed into ``_`` so that
    IDs like ``EG/2021/4385`` cannot create nested directories under
    ``data/enrolled_faces/``.
    """
    cleaned = _STUDENT_ID_SAFE_RE.sub("_", raw.strip()).strip("_")
    return cleaned or "unknown"


def _prompt_student_details() -> Tuple[str, str, bool]:
    """Collect student name and ID interactively and warn on duplicates.

    Returns ``(student_id, name, override)`` where ``override`` is True if
    the user chose to re-enroll an existing student.
    """
    name = input("Student full name: ").strip()
    raw_student_id = input("Student ID: ").strip()
    if not name or not raw_student_id:
        raise ValueError("Name and Student ID are required")

    student_id = _sanitize_student_id(raw_student_id)
    if student_id != raw_student_id:
        logger.info("Sanitized student ID %r -> %r", raw_student_id, student_id)
        print(f"Note: student ID normalised to '{student_id}' for safe storage.")

    override = False
    database.init_db()
    existing = {row["student_id"] for row in database.get_all_students()}
    if student_id in existing:
        logger.warning("Student %s is already enrolled.", student_id)
        answer = input(
            f"Student {student_id} is already enrolled. Re-enroll anyway? [y/N]: "
        ).strip().lower()
        if answer not in {"y", "yes"}:
            raise SystemExit("Enrollment aborted by user.")
        override = True

    return student_id, name, override


def _resolve_student_details(
    name: str | None, student_id: str | None
) -> Tuple[str, str, bool]:
    """Resolve student details from explicit args or interactive prompts.

    When both ``name`` and ``student_id`` are provided (e.g. from the web
    form), the interactive prompt is skipped. If the student already
    exists, re-enrollment proceeds automatically (override=True) instead
    of asking on the console.
    """
    if name and student_id:
        clean_name = name.strip()
        clean_id = _sanitize_student_id(student_id)
        if not clean_name or not clean_id:
            raise ValueError("Name and Student ID are required")
        database.init_db()
        existing = {row["student_id"] for row in database.get_all_students()}
        override = clean_id in existing
        if override:
            logger.info("Student %s already enrolled; re-enrolling.", clean_id)
        return clean_id, clean_name, override
    return _prompt_student_details()


def run_enrollment(
    name: str | None = None, student_id: str | None = None
) -> None:
    """Drive the enrollment session end-to-end.

    Args:
        name: Optional student full name. When provided together with
            ``student_id``, the interactive console prompt is skipped
            (used by the web form launch path).
        student_id: Optional student ID.
    """
    ensure_dirs()
    student_id, name, override = _resolve_student_details(name, student_id)

    mp_detector = build_mediapipe_detector()
    haar_classifier = build_haar_classifier()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {config.CAMERA_INDEX}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    window = "VisionGate Enrollment"
    cv2.namedWindow(window)

    raw_crops: List[np.ndarray] = []
    training_images: List[np.ndarray] = []
    captured = 0
    flash_frames_left = 0
    frame_counter = 0

    target = config.ENROLLMENT_IMAGES
    logger.info("Starting enrollment for %s (%s), target=%d", student_id, name, target)

    try:
        while captured < target:
            ok, raw_frame = cap.read()
            if not ok:
                logger.warning("Camera read failed; retrying")
                continue
            frame_counter += 1

            color_frame, _ = preprocess.full_preprocess_pipeline(raw_frame)
            display = color_frame.copy()

            should_capture = frame_counter % config.FRAME_SKIP == 0
            boxes: List[Tuple[int, int, int, int]] = []
            if should_capture:
                boxes = detect_faces(color_frame, mp_detector, haar_classifier)

            if boxes:
                x, y, w, h = boxes[0]
                cv2.rectangle(display, (x, y), (x + w, y + h), DETECTION_BOX_COLOR_BGR, 2)

                if should_capture and captured < target:
                    crop = crop_face_roi(color_frame, (x, y, w, h), padding=20)
                    if crop.size > 0:
                        raw_crops.append(crop)
                        training_images.extend(preprocess.augment_image(crop))
                        captured += 1
                        flash_frames_left = 2

            instruction_idx = (captured // INSTRUCTION_INTERVAL) % len(INSTRUCTIONS)
            _overlay_text(display, INSTRUCTIONS[instruction_idx], (10, 30))
            _overlay_text(display, f"Captured: {captured} / {target}", (10, 60))
            _overlay_text(display, "Press Q to abort", (10, display.shape[0] - 15))

            if flash_frames_left > 0:
                display = _flash(display)
                flash_frames_left -= 1

            cv2.imshow(window, display)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                logger.warning("Enrollment aborted by user at %d/%d captures", captured, target)
                raise SystemExit("Enrollment aborted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            mp_detector.close()
        except Exception:
            pass

    logger.info(
        "Captured %d raw frames, %d total training images (augmented)",
        len(raw_crops),
        len(training_images),
    )

    encodings, failures = compute_encodings(training_images)
    if not encodings:
        raise RuntimeError("No encodings could be computed — enrollment failed")

    labels = [student_id] * len(encodings)
    save_encodings(encodings, labels, config.ENCODINGS_PATH, names={student_id: name})

    label_map = _load_lbph_label_map()
    if student_id not in label_map:
        label_map[student_id] = _next_lbph_label(label_map)
    int_label = label_map[student_id]
    train_lbph(training_images, [int_label] * len(training_images), config.LBPH_MODEL_PATH)
    _save_lbph_label_map(label_map)

    if not override:
        database.register_student(student_id, name)

    student_dir = _save_raw_faces(student_id, raw_crops)

    print()
    print("=" * 60)
    print("Enrollment summary")
    print("=" * 60)
    print(f"  Student name           : {name}")
    print(f"  Student ID             : {student_id}")
    print(f"  Raw images captured    : {len(raw_crops)}")
    print(f"  Augmented training imgs: {len(training_images)}")
    print(f"  Encodings stored       : {len(encodings)}")
    print(f"  Encoding failures      : {failures}")
    print(f"  Raw images saved to    : {student_dir}")
    print(f"  Encodings file         : {config.ENCODINGS_PATH}")
    print(f"  LBPH model file        : {config.LBPH_MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VisionGate enrollment")
    parser.add_argument("--name", help="Student full name (skips prompt)")
    parser.add_argument("--id", dest="student_id", help="Student ID (skips prompt)")
    args = parser.parse_args()
    run_enrollment(name=args.name, student_id=args.student_id)
