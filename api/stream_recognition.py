"""In-process recognition that streams annotated frames to the web UI.

Same detection/encoding/liveness/attendance logic as recognize.py, but instead
of an OpenCV window it runs in a background thread and keeps the latest
annotated frame as a JPEG for the FastAPI MJPEG endpoint to serve. The native
recognize.py is left alone as a reliable fallback.
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
        # Cached heavy components (shared across sessions when pre-warmed).
        self._mp_detector = None
        self._haar_clf = None
        self._liveness_template: Optional[LivenessDetector] = None

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

    def prewarm(self) -> None:
        """Load heavy models once so the first session start is fast.

        Imports face_recognition and builds the MediaPipe detector + dlib
        liveness predictor ahead of time, caching them for reuse. Safe to
        call at API startup; failures are logged but non-fatal.
        """
        try:
            import face_recognition

            if self._mp_detector is None:
                self._mp_detector = build_mediapipe_detector()
            if self._haar_clf is None:
                self._haar_clf = build_haar_classifier()
            if self._liveness_template is None:
                self._liveness_template = LivenessDetector()
            logger.info("Recognition models pre-warmed")
        except Exception as exc:
            logger.warning("Pre-warm failed (will load lazily): %s", exc)

    # -- worker ----------------------------------------------------------

    def _run(self) -> None:
        import face_recognition  # heavy import deferred to thread

        self._running = True
        try:
            ensure_dirs()
            database.init_db()

            from recognize import _load_encodings, _load_lbph_model  # reuse loaders

            known_encodings, known_labels, name_map = _load_encodings()
            if not known_encodings:
                self._error = "No enrolled students yet. Enroll someone first."
                with self._lock:
                    self._latest_jpeg = _placeholder("No enrolled students")
                logger.warning(self._error)
                return

            _load_lbph_model()  # loaded for parity; matching uses encodings

            if self._mp_detector is None:
                self._mp_detector = build_mediapipe_detector()
            if self._haar_clf is None:
                self._haar_clf = build_haar_classifier()
            mp_detector = self._mp_detector
            haar_clf = self._haar_clf

            cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
            if not cap.isOpened():
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

            # Shared state between the capture loop and the recognition thread.
            latest = {"frame": None}
            frame_lock = threading.Lock()
            results_lock = threading.Lock()
            results: list[dict] = []  # most recent recognition overlays

            def recognition_worker() -> None:
                """Heavy detection/encoding on the latest frame only.

                A spatial cache avoids re-encoding a face that was recognized
                a moment ago and is still roughly in the same place — the main
                throughput win when people stand in front of the camera.
                """
                liveness_by_student: dict[str, LivenessDetector] = {}
                cache: list[dict] = []
                CACHE_TTL = 3.0  # seconds a cached recognition stays valid

                def _match_cache(cx: float, cy: float, w: float):
                    now = time.time()
                    for c in cache:
                        if now - c["t"] > CACHE_TTL:
                            continue
                        if abs(c["center"][0] - cx) < w * 0.6 and abs(
                            c["center"][1] - cy
                        ) < w * 0.6:
                            return c
                    return None

                while not self._stop_event.is_set():
                    with frame_lock:
                        frame = None if latest["frame"] is None else latest["frame"].copy()
                    if frame is None:
                        time.sleep(0.01)
                        continue

                    color_frame, _gray = preprocess.full_preprocess_pipeline(frame)
                    faces = detect_faces(color_frame, mp_detector, haar_clf)

                    now = time.time()
                    cache[:] = [c for c in cache if now - c["t"] <= CACHE_TTL]

                    new_results: list[dict] = []
                    for bbox in faces:
                        x, y, w, h = bbox
                        cx, cy = x + w / 2.0, y + h / 2.0
                        cached = _match_cache(cx, cy, w)
                        if cached is not None and cached["confirmed"]:
                            new_results.append(
                                {
                                    "bbox": bbox,
                                    "label": cached["name"],
                                    "confidence": cached["confidence"],
                                    "status": "recognized",
                                }
                            )
                            cached["center"] = (cx, cy)
                            cached["t"] = now
                            continue

                        face_roi = crop_face_roi(color_frame, bbox)
                        if face_roi.size == 0:
                            continue
                        rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                        encs = face_recognition.face_encodings(rgb_roi)
                        if not encs:
                            new_results.append(
                                {"bbox": bbox, "label": "No Encoding", "confidence": 0.0, "status": "unknown"}
                            )
                            continue

                        distances = face_recognition.face_distance(known_encodings, encs[0])
                        best_idx = int(np.argmin(distances))
                        best_dist = float(distances[best_idx])
                        confidence = max(0.0, 1.0 - best_dist)

                        if best_dist <= config.RECOGNITION_THRESHOLD:
                            student_id = known_labels[best_idx]
                            name = name_map.get(student_id, student_id)
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
                                cache.append(
                                    {
                                        "center": (cx, cy),
                                        "label": student_id,
                                        "name": name,
                                        "status": "recognized",
                                        "confidence": confidence,
                                        "t": now,
                                        "confirmed": True,
                                    }
                                )
                                new_results.append(
                                    {"bbox": bbox, "label": name, "confidence": confidence, "status": "recognized"}
                                )
                            else:
                                new_results.append(
                                    {"bbox": bbox, "label": "Blink to verify", "confidence": 0.0, "status": "spoof"}
                                )
                        else:
                            new_results.append(
                                {"bbox": bbox, "label": "Unknown", "confidence": confidence, "status": "unknown"}
                            )

                    with results_lock:
                        results[:] = new_results

            rec_thread = threading.Thread(
                target=recognition_worker, name="recognition-worker", daemon=True
            )
            rec_thread.start()

            # Capture + stream loop stays fast because it never does encoding.
            start_time = time.time()
            frame_count = 0
            try:
                while not self._stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("Camera read failed")
                        time.sleep(0.05)
                        continue

                    frame = preprocess.resize_frame(
                        frame, config.FRAME_WIDTH, config.FRAME_HEIGHT
                    )
                    with frame_lock:
                        latest["frame"] = frame

                    frame_count += 1
                    with results_lock:
                        current = list(results)
                    for r in current:
                        draw_bounding_box(
                            frame, r["bbox"], r["label"], r["confidence"], r["status"]
                        )
                    self._overlay_status(frame, start_time, frame_count)
                    self._set_jpeg(_encode_jpeg(frame))
                    time.sleep(0.005)
            finally:
                self._stop_event.set()
                rec_thread.join(timeout=3)
                cap.release()
                logger.info("Streaming session %s ended", self._session)
                try:
                    database.export_to_csv(datetime.now().strftime("%Y-%m-%d"))
                except Exception as exc:
                    logger.error("CSV export failed: %s", exc)
        except Exception as exc:
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
