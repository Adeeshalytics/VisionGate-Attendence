"""Enroll students from folders of photos instead of the live webcam.

Handy when classmates aren't around for a live capture - drop a few photos
of each person into a folder and enroll everyone at once. Layout is one
sub-folder per person:

    people/
        Adeesha M.G.P/
            img1.jpg
        EG2021999_Jane/
            a.jpg

The folder name becomes the display name and a safe slug of it becomes the
student ID; override both for a single folder with --name / --id. Uses the
same detection/encoding/LBPH path as enroll.py so the gallery stays compatible.

    python enroll_from_images.py people/
    python enroll_from_images.py people/Adeesha --name "Adeesha M.G.P" --id EG2021001
    python enroll_from_images.py people/ --no-augment
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np

import config
import database
from preprocess import augment_image
from utils import ensure_dirs, get_logger
from enroll import (
    _load_lbph_label_map,
    _next_lbph_label,
    _sanitize_student_id,
    _save_lbph_label_map,
    _save_raw_faces,
    build_haar_classifier,
    build_mediapipe_detector,
    compute_encodings,
    crop_face_roi,
    detect_faces,
    save_encodings,
    train_lbph,
)

logger = get_logger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_image_paths(folder: Path) -> List[Path]:
    """Return all image files directly inside ``folder`` (sorted)."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )


def _crops_from_image(
    img_path: Path, mp_detector, haar_clf, padding: int = 20
) -> List[np.ndarray]:
    """Detect faces in one photo and return padded BGR crops.

    Falls back to an empty list (with a warning) if the image can't be read
    or no face is found.
    """
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        logger.warning("Could not read image %s", img_path)
        return []

    boxes = detect_faces(bgr, mp_detector, haar_clf)
    if not boxes:
        logger.warning("No face detected in %s (skipped)", img_path.name)
        return []

    crops: List[np.ndarray] = []
    for bbox in boxes:
        crop = crop_face_roi(bgr, bbox, padding=padding)
        if crop.size > 0:
            crops.append(crop)
    return crops


def enroll_person(
    folder: Path,
    mp_detector,
    haar_clf,
    name: str | None = None,
    student_id: str | None = None,
    augment: bool = True,
) -> dict:
    """Enroll one person from all photos in ``folder``.

    Returns a summary dict; raises nothing fatal — folders with no usable
    faces are reported with ``enrolled=False``.
    """
    display_name = (name or folder.name).strip()
    sid = _sanitize_student_id(student_id or folder.name)

    image_paths = _collect_image_paths(folder)
    if not image_paths:
        logger.warning("No images found in %s", folder)
        return {"student_id": sid, "name": display_name, "enrolled": False,
                "reason": "no images", "raw": 0, "encodings": 0}

    raw_crops: List[np.ndarray] = []
    training_images: List[np.ndarray] = []
    for img_path in image_paths:
        for crop in _crops_from_image(img_path, mp_detector, haar_clf):
            raw_crops.append(crop)
            if augment:
                training_images.extend(augment_image(crop))
            else:
                training_images.append(crop)

    if not raw_crops:
        return {"student_id": sid, "name": display_name, "enrolled": False,
                "reason": "no faces detected", "raw": 0, "encodings": 0}

    encodings, failures = compute_encodings(training_images)
    if not encodings:
        return {"student_id": sid, "name": display_name, "enrolled": False,
                "reason": "no encodings could be computed", "raw": len(raw_crops),
                "encodings": 0}

    # Persist exactly like the webcam path.
    save_encodings(encodings, [sid] * len(encodings), config.ENCODINGS_PATH,
                   names={sid: display_name})

    label_map = _load_lbph_label_map()
    if sid not in label_map:
        label_map[sid] = _next_lbph_label(label_map)
    train_lbph(training_images, [label_map[sid]] * len(training_images),
               config.LBPH_MODEL_PATH)
    _save_lbph_label_map(label_map)

    database.register_student(sid, display_name)
    _save_raw_faces(sid, raw_crops)

    return {
        "student_id": sid,
        "name": display_name,
        "enrolled": True,
        "images": len(image_paths),
        "raw": len(raw_crops),
        "encodings": len(encodings),
        "encoding_failures": failures,
    }


def _person_folders(root: Path, single: bool) -> List[Path]:
    """Return the list of person folders to process."""
    if single:
        return [root]
    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    # If the root itself contains images (and no sub-folders), treat it as one person.
    if not subdirs and _collect_image_paths(root):
        return [root]
    return subdirs


def run(root: Path, name: str | None, student_id: str | None,
        augment: bool) -> List[dict]:
    """Enroll everyone found under ``root`` and return per-person summaries."""
    ensure_dirs()
    database.init_db()

    single = bool(name or student_id)
    folders = _person_folders(root, single)
    if not folders:
        raise SystemExit(f"No person folders or images found under {root}")

    mp_detector = build_mediapipe_detector()
    haar_clf = build_haar_classifier()

    summaries: List[dict] = []
    try:
        for folder in folders:
            logger.info("Enrolling from %s", folder)
            summaries.append(
                enroll_person(folder, mp_detector, haar_clf,
                              name=name, student_id=student_id, augment=augment)
            )
    finally:
        try:
            mp_detector.close()
        except Exception:
            pass
    return summaries


def _print_summary(summaries: List[dict]) -> None:
    ok = [s for s in summaries if s.get("enrolled")]
    print("\n" + "=" * 64)
    print("Enroll-from-images summary")
    print("=" * 64)
    for s in summaries:
        if s.get("enrolled"):
            print(f"  [OK]   {s['name']} ({s['student_id']}): "
                  f"{s['images']} imgs -> {s['raw']} faces -> "
                  f"{s['encodings']} encodings ({s['encoding_failures']} failed)")
        else:
            print(f"  [SKIP] {s['name']} ({s['student_id']}): {s.get('reason')}")
    print("-" * 64)
    print(f"  Enrolled {len(ok)}/{len(summaries)} people")
    print(f"  Encodings file : {config.ENCODINGS_PATH}")
    print(f"  LBPH model     : {config.LBPH_MODEL_PATH}")
    print("=" * 64)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Enroll students from photos")
    parser.add_argument("root", type=Path,
                        help="Folder of per-person sub-folders, OR a single person's folder")
    parser.add_argument("--name", help="Display name (treats root as one person)")
    parser.add_argument("--id", dest="student_id",
                        help="Student ID (treats root as one person)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable augmentation (use raw crops only)")
    args = parser.parse_args(argv)

    summaries = run(args.root, args.name, args.student_id, augment=not args.no_augment)
    _print_summary(summaries)


if __name__ == "__main__":
    main()
