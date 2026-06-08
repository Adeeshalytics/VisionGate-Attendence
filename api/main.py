"""FastAPI backend for the VisionGate web frontend.

This is a thin REST layer over the existing :mod:`database` module. It does
**not** touch the camera pipeline — enrollment and recognition keep running
as native Python processes that write to the same SQLite database. The API
only reads/serves that data (plus a couple of helper endpoints to launch the
native scripts and export CSVs).

Run with::

    python -m uvicorn api.main:app --reload --port 8000

from the project root, inside the ``visiongate`` conda environment.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Make the project root importable so we can reuse the existing modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
import database  # noqa: E402
from utils import get_logger  # noqa: E402

logger = get_logger("api")

# Handle to the native recognition window process (recognize.py), so the UI
# can start and stop it. Only one runs at a time.
_recognize_proc: Optional[subprocess.Popen] = None

app = FastAPI(
    title="VisionGate API",
    description="REST layer over the VisionGate attendance database.",
    version="1.0.0",
)

# The Next.js dev server runs on :3000; allow it (and common localhost ports)
# to call the API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic response/request models
# ---------------------------------------------------------------------------

class Student(BaseModel):
    student_id: str
    name: str
    enrolled_at: Optional[str] = None


class AttendanceRecord(BaseModel):
    id: Optional[int] = None
    student_id: str
    name: str
    date: str
    time: str
    confidence: Optional[float] = None
    session: Optional[str] = None


class TodayOverview(BaseModel):
    date: str
    present_count: int
    total_enrolled: int
    attendance_rate: float
    records: list[AttendanceRecord]


class StudentSummary(BaseModel):
    student_id: str
    name: str
    total_sessions: int
    enrolled_at: Optional[str] = None


class AnalyticsResponse(BaseModel):
    total_students: int
    total_sessions_held: int
    leaderboard: list[StudentSummary]
    daily_counts: list[dict]
    low_attendance: list[dict]


class LaunchResponse(BaseModel):
    started: bool
    message: str
    pid: Optional[int] = None


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def _startup() -> None:
    database.init_db()
    logger.info("VisionGate API started; DB at %s", config.DB_PATH)


# ---------------------------------------------------------------------------
# Health / meta
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "db_path": str(config.DB_PATH)}


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------

@app.get("/students", response_model=list[Student])
def list_students() -> list[dict]:
    students = database.get_all_students()
    return [
        {
            "student_id": s["student_id"],
            "name": s["name"],
            "enrolled_at": str(s.get("enrolled_at")) if s.get("enrolled_at") else None,
        }
        for s in students
    ]


@app.delete("/students")
def delete_all_students() -> dict:
    """Delete ALL students plus their encodings, models and face crops.

    Refuses while a live recognition session is running, since that session
    holds the camera and reads the encodings being deleted.
    """
    from api.stream_recognition import streamer as _streamer

    if _streamer.running:
        raise HTTPException(
            status_code=409,
            detail="Stop the live session before deleting all students.",
        )
    from api.reset_all import reset_all

    summary = reset_all(verbose=False)
    logger.info("Deleted all students via API: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Attendance
# ---------------------------------------------------------------------------

def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


@app.get("/attendance/today", response_model=TodayOverview)
def attendance_today() -> dict:
    today = _today_str()
    records = database.get_attendance_by_date(today)
    students = database.get_all_students()
    total_enrolled = len(students)
    present_ids = {r["student_id"] for r in records}
    present_count = len(present_ids)
    rate = (present_count / total_enrolled * 100.0) if total_enrolled else 0.0
    return {
        "date": today,
        "present_count": present_count,
        "total_enrolled": total_enrolled,
        "attendance_rate": round(rate, 1),
        "records": records,
    }


@app.get("/attendance/{day}", response_model=list[AttendanceRecord])
def attendance_by_date(day: str) -> list[dict]:
    """Return attendance records for ``day`` (format ``YYYY-MM-DD``)."""
    try:
        datetime.strptime(day, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")
    return database.get_attendance_by_date(day)


@app.delete("/attendance/{day}")
def clear_attendance(day: str) -> dict:
    """Delete all attendance records for ``day`` (keeps enrolled students)."""
    try:
        datetime.strptime(day, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")
    removed = database.delete_attendance_by_date(day)
    # Keep the live "Recognized this session" sidebar in sync: if we cleared
    # today's records, also clear the streamer's in-memory list so it stops
    # showing students that no longer have attendance rows.
    if day == _today_str():
        try:
            from api.stream_recognition import streamer as _streamer

            _streamer.reset_recognized()
        except Exception:  # noqa: BLE001
            pass
    return {"date": day, "removed": removed}


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

@app.get("/analytics", response_model=AnalyticsResponse)
def analytics(
    days: int = Query(14, ge=1, le=90),
    low_threshold: float = Query(0.8, ge=0.0, le=1.0),
) -> dict:
    summary = database.get_attendance_summary()
    students = database.get_all_students()
    student_name = {s["student_id"]: s["name"] for s in students}
    enrolled_at = {s["student_id"]: s.get("enrolled_at") for s in students}

    # Leaderboard: students by sessions attended (desc)
    leaderboard = sorted(
        (
            {
                "student_id": sid,
                "name": info.get("name") or student_name.get(sid, sid),
                "total_sessions": info.get("total_sessions", 0),
                "enrolled_at": str(enrolled_at.get(sid)) if enrolled_at.get(sid) else None,
            }
            for sid, info in summary.items()
        ),
        key=lambda r: r["total_sessions"],
        reverse=True,
    )

    # Distinct session days held across all students
    all_dates: set[str] = set()
    for info in summary.values():
        all_dates.update(info.get("dates", []))
    total_sessions_held = len(all_dates)

    # Daily counts for the last ``days`` days
    daily_counts = []
    for i in range(days - 1, -1, -1):
        d = (date.today() - timedelta(days=i)).strftime("%Y-%m-%d")
        daily_counts.append({"date": d, "count": len(database.get_attendance_by_date(d))})

    # Students below the attendance threshold
    low_attendance = []
    if total_sessions_held > 0:
        for s in students:
            sid = s["student_id"]
            attended = summary.get(sid, {}).get("total_sessions", 0)
            rate = attended / total_sessions_held
            if rate < low_threshold:
                low_attendance.append(
                    {
                        "student_id": sid,
                        "name": s["name"],
                        "sessions_attended": attended,
                        "sessions_held": total_sessions_held,
                        "attendance_rate": round(rate * 100, 1),
                    }
                )
        low_attendance.sort(key=lambda r: r["attendance_rate"])

    return {
        "total_students": len(students),
        "total_sessions_held": total_sessions_held,
        "leaderboard": leaderboard,
        "daily_counts": daily_counts,
        "low_attendance": low_attendance,
    }


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

@app.get("/export/{day}")
def export_csv(day: str) -> FileResponse:
    try:
        datetime.strptime(day, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")
    path: Path = database.export_to_csv(day)
    if not path.exists():
        raise HTTPException(status_code=500, detail="CSV could not be created")
    return FileResponse(
        path,
        media_type="text/csv",
        filename=path.name,
    )


# ---------------------------------------------------------------------------
# Launch native camera scripts (enrollment / recognition)
# ---------------------------------------------------------------------------

def _launch_script(script: str, args: list[str] | None = None, new_console: bool = True) -> LaunchResponse:
    script_path = PROJECT_ROOT / script
    if not script_path.exists():
        raise HTTPException(status_code=404, detail=f"{script} not found")

    cmd = [sys.executable, str(script_path)] + (args or [])

    # When a script still needs console input we give it its own terminal
    # (CREATE_NEW_CONSOLE). When all inputs are passed as args, we launch it
    # silently (no console) — only its OpenCV webcam window will appear.
    kwargs: dict = {"cwd": str(PROJECT_ROOT)}
    create_new_console = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    if new_console and create_new_console:
        kwargs["creationflags"] = create_new_console
    else:
        kwargs["stdin"] = subprocess.DEVNULL

    try:
        proc = subprocess.Popen(cmd, **kwargs)
    except OSError as exc:  # noqa: BLE001
        logger.exception("Failed to launch %s", script)
        raise HTTPException(status_code=500, detail=str(exc))
    logger.info("Launched %s (pid=%s)", script, proc.pid)
    return LaunchResponse(
        started=True,
        message=f"{script} launched — the webcam window will open shortly.",
        pid=proc.pid,
    )


class EnrollRequest(BaseModel):
    name: str
    student_id: str


@app.post("/launch/enroll", response_model=LaunchResponse)
def launch_enroll(req: EnrollRequest) -> LaunchResponse:
    """Launch enrollment for a named student (opens only a webcam window).

    Name and ID come from the web form, so no console prompt is needed and
    the script runs without a terminal window.
    """
    name = req.name.strip()
    student_id = req.student_id.strip()
    if not name or not student_id:
        raise HTTPException(status_code=400, detail="name and student_id are required")
    return _launch_script(
        "enroll.py",
        args=["--name", name, "--id", student_id],
        new_console=False,
    )


@app.post("/launch/recognize", response_model=LaunchResponse)
def launch_recognize() -> LaunchResponse:
    """Launch the native recognition session (opens a webcam window)."""
    global _recognize_proc
    if _recognize_proc is not None and _recognize_proc.poll() is None:
        return LaunchResponse(
            started=False,
            message="A recognition window is already running.",
            pid=_recognize_proc.pid,
        )

    script_path = PROJECT_ROOT / "recognize.py"
    if not script_path.exists():
        raise HTTPException(status_code=404, detail="recognize.py not found")
    try:
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
        )
    except OSError as exc:  # noqa: BLE001
        logger.exception("Failed to launch recognize.py")
        raise HTTPException(status_code=500, detail=str(exc))

    _recognize_proc = proc
    logger.info("Launched recognize.py (pid=%s)", proc.pid)
    return LaunchResponse(
        started=True,
        message="Recognition window launched — it will open shortly.",
        pid=proc.pid,
    )


@app.post("/launch/recognize/stop", response_model=LaunchResponse)
def stop_recognize() -> LaunchResponse:
    """Terminate the native recognition window if it is running."""
    global _recognize_proc
    if _recognize_proc is None or _recognize_proc.poll() is not None:
        _recognize_proc = None
        return LaunchResponse(started=False, message="No recognition window is running.")
    pid = _recognize_proc.pid
    try:
        _recognize_proc.terminate()
        try:
            _recognize_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _recognize_proc.kill()
    except OSError as exc:  # noqa: BLE001
        logger.warning("Failed to stop recognize.py: %s", exc)
    finally:
        _recognize_proc = None
    logger.info("Stopped recognize.py (pid=%s)", pid)
    return LaunchResponse(started=False, message="Recognition window stopped.", pid=pid)


@app.get("/launch/recognize/status")
def recognize_status() -> dict:
    """Report whether the native recognition window is currently running."""
    running = _recognize_proc is not None and _recognize_proc.poll() is None
    return {"running": running, "pid": _recognize_proc.pid if running else None}


# ---------------------------------------------------------------------------
# In-dashboard recognition stream (MJPEG)
# ---------------------------------------------------------------------------

import time  # noqa: E402

from api.stream_recognition import streamer  # noqa: E402


class RecognizedStudent(BaseModel):
    student_id: str
    name: str
    time: str
    confidence: Optional[float] = None


class StreamStatus(BaseModel):
    running: bool
    session: str
    error: Optional[str] = None
    recognized_count: int
    students: list[RecognizedStudent] = []


@app.post("/stream/start")
def stream_start() -> dict:
    """Start the in-process recognition session that streams to the browser."""
    return streamer.start()


@app.post("/stream/stop")
def stream_stop() -> dict:
    """Stop the in-dashboard recognition stream and release the camera."""
    return streamer.stop()


@app.get("/stream/status", response_model=StreamStatus)
def stream_status() -> dict:
    return streamer.status()


def _mjpeg_generator():
    boundary = b"--frame"
    while True:
        jpeg = streamer.get_jpeg()
        if jpeg:
            yield (
                boundary
                + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(jpeg)).encode()
                + b"\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
        # ~20 fps ceiling for the HTTP push; the worker may produce fewer.
        time.sleep(0.05)
        if not streamer.running and not jpeg:
            break


@app.get("/stream/video")
def stream_video() -> StreamingResponse:
    """MJPEG stream of the annotated recognition feed."""
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
