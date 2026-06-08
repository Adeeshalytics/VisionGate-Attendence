"""In-process recognition session that streams annotated frames to the web UI.

This reuses the exact same detection / encoding / liveness / attendance
logic as :mod:`recognize`, but instead of showing an OpenCV window
(``cv2.imshow``) it runs the loop in a background thread and keeps the most
recent annotated frame as a JPEG that the FastAPI MJPEG endpoint can serve.

The native ``recognize.py`` script is left untouched as a reliable fallback.
"""

from __future__ import annotations

import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
import database  # noqa: E402
import preprocess  # noqa: E402
from anti_spoof import LivenessDetector  # noqa: E402
from enroll import (  # noqa: E402
    build_haar_classifier,
    build_mediapipe_detector,
    crop_face_roi,
    detect_faces,
)
from utils import draw_bounding_box, ensure_dirs, get_logger, get_session_id  # noqa: E402

logger = get_logger("stream")


def _encode_jpeg(frame: np.ndarray, quality: int = 70) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return b""
    return buf.tobytes()


def _placeholder(text: str) -> bytes:
    img = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    img[:] = (24, 18, 14)  # dark slate (BGR)
    cv2.putText(
        img,
        text,
        (30, config.FRAME_HEIGHT // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    return _encode_jpeg(img)


class RecognitionStreamer:
    """Singleton-style recognition session producing JPEG frames.

    Thread model: a single worker thread owns the webcam and runs the
    recognition loop, writing the latest annotated JPEG under a lock.
    HTTP request handlers read that JPEG without touching the camera.
    """

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_jpeg: bytes = _placeholder("Camera not started")
        self._running = False
        self._session: str = ""
        self._error: Optional[str] = None
        self._recognized: dict[str, dict] = {}

    # -- lifecycle -------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    def status(self) -> dict:
        with self._lock:
            students = [
                {
                    "student_id": sid,
                    "name": info["name"],
                    "time": info["time"],
                    "confidence": info["confidence"],
                }
                for sid, info in self._recognized.items()
            ]
        # Most recent first
        students.sort(key=lambda s: s["time"], reverse=True)
        return {
            "running": self._running,
            "session": self._session,
            "error": self._error,
            "recognized_count": len(students),
            "students": students,
        }

    def start(self) -> dict:
        if self._running:
            return {"started": False, "message": "A session is already running."}

        self._stop_event.clear()
        self._error = None
        self._recognized = {}
        self._session = get_session_id()
        self._thread = threading.Thread(target=self._run, name="recognition", daemon=True)
        self._thread.start()
        # Give the worker a brief moment to open the camera / report errors.
        time.sleep(0.5)
        if self._error:
            return {"started": False, "message": self._error}
        return {"started": True, "message": "Recognition session started.", "session": self._session}

    def stop(self) -> dict:
        if not self._running and self._thread is None:
            return {"stopped": True, "message": "No session running."}
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None
        with self._lock:
            self._latest_jpeg = _placeholder("Session ended")
        return {"stopped": True, "message": "Recognition session stopped."}

    # -- frame access ----------------------------------------------------

    def get_jpeg(self) -> bytes:
        with self._lock:
            return self._latest_jpeg

    def _set_jpeg(self, jpeg: bytes) -> None:
        if jpeg:
            with self._lock:
                self._latest_jpeg = jpeg

    def reset_recognized(self) -> None:
        """Clear the in-memory list of recognized students for this session.

        Used when attendance for the active session's day is cleared, so the
        sidebar stays in sync with the (now-empty) database.
        """
        with self._lock:
            self._recognized = {}

    # -- worker ----------------------------------------------------------

    def _run(self) -> None:
        import face_recognition  # heavy import deferred to thread

        self._running = True
        try:
            ensure_dirs()
            database.init_db()

            # Load enrolled data
            from recognize import _load_encodings, _load_lbph_model  # reuse loaders

            known_encodings, known_labels, name_map = _load_encodings()
            if not known_encodings:
                self._error = "No enrolled students yet. Enroll someone first."
                with self._lock:
                    self._latest_jpeg = _placeholder("No enrolled students")
                logger.warning(self._error)
                return

            _load_lbph_model()  # loaded for parity; matching uses encodings

            mp_detector = build_mediapipe_detector()
            haar_clf = build_haar_classifier()
            # Per-student liveness detectors: each recognized student gets
            # their own blink-tracking state so multiple people in frame don't
            # share (and corrupt) one another's blink counters.
            liveness_by_student: dict[str, LivenessDetector] = {}

            cap = cv2.VideoCapture(config.CAMERA_INDEX)
            if not cap.isOpened():
                self._error = f"Could not open camera at index {config.CAMERA_INDEX}"
                with self._lock:
                    self._latest_jpeg = _placeholder("Camera unavailable")
                logger.error(self._error)
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

            logger.info("Streaming recognition session %s started", self._session)
            frame_count = 0
            start_time = time.time()

            try:
                while not self._stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("Camera read failed")
                        time.sleep(0.05)
                        continue

                    frame_count += 1
                    # Skip heavy processing on some frames but still stream them.
                    if frame_count % config.FRAME_SKIP != 0:
                        self._overlay_status(frame, start_time, frame_count)
                        self._set_jpeg(_encode_jpeg(frame))
                        continue

                    color_frame, gray_frame = preprocess.full_preprocess_pipeline(frame)
                    faces = detect_faces(color_frame, mp_detector, haar_clf)

                    for bbox in faces:
                        face_roi_color = crop_face_roi(color_frame, bbox)
                        if face_roi_color.size == 0:
                            continue

                        rgb_roi = cv2.cvtColor(face_roi_color, cv2.COLOR_BGR2RGB)
                        encs = face_recognition.face_encodings(rgb_roi)
                        if not encs:
                            draw_bounding_box(frame, bbox, "No Encoding", 0.0, "unknown")
                            continue

                        distances = face_recognition.face_distance(known_encodings, encs[0])
                        best_idx = int(np.argmin(distances))
                        best_dist = float(distances[best_idx])
                        confidence = max(0.0, 1.0 - best_dist)

                        if best_dist <= config.RECOGNITION_THRESHOLD:
                            student_id = known_labels[best_idx]
                            name = name_map.get(student_id, student_id)
                            # Get-or-create this student's own liveness tracker
                            # so each person's blinks are counted independently.
                            detector = liveness_by_student.get(student_id)
                            if detector is None:
                                detector = LivenessDetector()
                                liveness_by_student[student_id] = detector
                            verdict = detector.check(color_frame, bbox)
                            if verdict["is_live"]:
                                marked = database.mark_attendance(
                                    student_id, name, confidence, self._session
                                )
                                if marked:
                                    logger.info("Attendance marked: %s", name)
                                with self._lock:
                                    if student_id not in self._recognized:
                                        self._recognized[student_id] = {
                                            "name": name,
                                            "time": datetime.now().strftime("%H:%M:%S"),
                                            "confidence": round(confidence, 3),
                                        }
                                draw_bounding_box(frame, bbox, name, confidence, "recognized")
                            else:
                                draw_bounding_box(frame, bbox, "Blink to verify", 0.0, "spoof")
                        else:
                            draw_bounding_box(frame, bbox, "Unknown", confidence, "unknown")

                    self._overlay_status(frame, start_time, frame_count)
                    self._set_jpeg(_encode_jpeg(frame))
            finally:
                cap.release()
                try:
                    mp_detector.close()
                except Exception:  # noqa: BLE001
                    pass
                logger.info("Streaming session %s ended", self._session)
                try:
                    database.export_to_csv(datetime.now().strftime("%Y-%m-%d"))
                except Exception as exc:  # noqa: BLE001
                    logger.error("CSV export failed: %s", exc)
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            logger.exception("Recognition stream crashed: %s", exc)
            with self._lock:
                self._latest_jpeg = _placeholder("Recognition error")
        finally:
            self._running = False

    def _overlay_status(self, frame: np.ndarray, start_time: float, frame_count: int) -> None:
        elapsed = max(1e-6, time.time() - start_time)
        fps = frame_count / elapsed
        text = f"FPS {fps:.1f} | Session {self._session} | Present {len(self._recognized)}"
        cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


# Module-level singleton shared by the API endpoints.
streamer = RecognitionStreamer()
