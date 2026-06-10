"""Real-time recognition: detect faces, match against enrolled embeddings,
check liveness, and write attendance. Run with `python recognize.py`.
"""

from __future__ import annotations

import pickle
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

import config
import database
import preprocess
from anti_spoof import LivenessDetector
from enroll import (
    build_haar_classifier,
    build_mediapipe_detector,
    crop_face_roi,
    detect_faces,
)
from utils import draw_bounding_box, ensure_dirs, get_logger, get_session_id

logger = get_logger(__name__)


def _load_encodings() -> Tuple[List[np.ndarray], List[str], dict[str, str]]:
    """Load encodings, labels and the id->name map. Returns empties if the
    pickle is missing or unreadable."""
    path: Path = config.ENCODINGS_PATH
    if not path.exists():
        logger.warning("Encodings file missing at %s — no students to recognize", path)
        return [], [], {}

    try:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
    except (pickle.UnpicklingError, EOFError) as exc:
        logger.error("Failed to read encodings at %s: %s", path, exc)
        return [], [], {}

    encodings = list(data.get("encodings", []))
    labels = list(data.get("labels", []))
    names = dict(data.get("names", {}))
    return encodings, labels, names


def _load_lbph_model() -> cv2.face_LBPHFaceRecognizer | None:
    path: Path = config.LBPH_MODEL_PATH
    if not path.exists():
        logger.warning("LBPH model missing at %s — continuing without it", path)
        return None
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(str(path))
        logger.info("Loaded LBPH model from %s", path)
        return recognizer
    except cv2.error as exc:
        logger.warning("Could not load LBPH model: %s", exc)
        return None


def _resolve_name(student_id: str, name_map: dict[str, str]) -> str:
    """Display name for a student id, falling back to a DB lookup, then the id."""
    if student_id in name_map:
        return name_map[student_id]
    try:
        for row in database.get_all_students():
            if row["student_id"] == student_id:
                name_map[student_id] = row["name"]
                return row["name"]
    except Exception as exc:
        logger.debug("DB name lookup failed for %s: %s", student_id, exc)
    return student_id


def _put_status_text(frame: np.ndarray, text: str, origin: Tuple[int, int]) -> None:
    cv2.putText(
        frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
    )


def run_recognition(session: str | None = None) -> None:
    """Run the recognition loop until the user presses Q. ``session`` defaults
    to the hour-rounded current time."""
    import face_recognition

    ensure_dirs()
    database.init_db()

    known_encodings, known_labels, name_map = _load_encodings()
    lbph = _load_lbph_model()

    unique_students = sorted(set(known_labels))
    logger.info(
        "Startup: %d enrolled student(s), %d total encoding(s), LBPH=%s",
        len(unique_students),
        len(known_encodings),
        "loaded" if lbph is not None else "unavailable",
    )

    if not known_encodings:
        logger.error("No encodings available — aborting recognition session")
        return

    session = session or get_session_id()
    logger.info("Recognition session: %s", session)

    mp_detector = build_mediapipe_detector()
    haar_clf = build_haar_classifier()
    liveness_detector = LivenessDetector()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {config.CAMERA_INDEX}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    window = "VisionGate -- Recognition"
    cv2.namedWindow(window)

    frame_count = 0
    session_log: dict[str, int] = defaultdict(int)
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Camera read failed")
                break

            frame_count += 1
            if frame_count % config.FRAME_SKIP != 0:
                cv2.imshow(window, frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                continue

            color_frame, _ = preprocess.full_preprocess_pipeline(frame)
            faces = detect_faces(color_frame, mp_detector, haar_clf)

            for bbox in faces:
                face_roi_color = crop_face_roi(color_frame, bbox)
                if face_roi_color.size == 0:
                    continue

                rgb_roi = cv2.cvtColor(face_roi_color, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_roi)
                if not encodings:
                    draw_bounding_box(frame, bbox, "No Encoding", 0.0, "unknown")
                    continue

                distances = face_recognition.face_distance(known_encodings, encodings[0])
                best_idx = int(np.argmin(distances))
                best_dist = float(distances[best_idx])
                confidence = max(0.0, 1.0 - best_dist)

                if best_dist > config.RECOGNITION_THRESHOLD:
                    draw_bounding_box(frame, bbox, "Unknown", confidence, "unknown")
                    continue

                student_id = known_labels[best_idx]
                name = _resolve_name(student_id, name_map)

                if not liveness_detector.check(color_frame, bbox)["is_live"]:
                    logger.warning("Spoof / no-blink detected for %s", name)
                    draw_bounding_box(frame, bbox, "Spoof", 0.0, "spoof")
                    continue

                if database.mark_attendance(student_id, name, confidence, session):
                    logger.info("Attendance marked: %s (%.2f%%)", name, confidence * 100.0)
                session_log[student_id] += 1
                draw_bounding_box(frame, bbox, name, confidence, "recognized")

            elapsed = max(1e-6, time.time() - start_time)
            fps = frame_count / elapsed
            _put_status_text(
                frame,
                f"FPS: {fps:.1f} | Session: {session} | Enrolled: {len(unique_students)}",
                (10, 30),
            )

            cv2.imshow(window, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            mp_detector.close()
        except Exception:
            pass

        logger.info("=== SESSION SUMMARY ===")
        logger.info("Session ID: %s", session)
        logger.info("Students recognized: %d", len(session_log))
        for sid, count in session_log.items():
            logger.info("  %s (%s): seen %d times", sid, name_map.get(sid, sid), count)

        try:
            csv_path = database.export_to_csv(datetime.now().strftime("%Y-%m-%d"))
            logger.info("Attendance exported to %s", csv_path)
        except Exception as exc:
            logger.error("CSV export failed: %s", exc)


if __name__ == "__main__":
    run_recognition()
