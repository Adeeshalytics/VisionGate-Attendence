"""SQLite persistence layer for VisionGate.

Stores enrolled students and attendance records. Reads ``config.DB_PATH``
at call time so tests can override the path between calls.
"""

from __future__ import annotations

import csv
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import config
from utils import get_logger

logger = get_logger(__name__)

_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection rooted at the current ``config.DB_PATH``."""
    db_path: Path = config.DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create the ``students`` and ``attendance`` tables if they don't exist."""
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                confidence REAL,
                session TEXT DEFAULT 'default',
                FOREIGN KEY (student_id) REFERENCES students(student_id)
            );
            """
        )
    logger.info("Database initialised at %s", config.DB_PATH)


def register_student(student_id: str, name: str) -> bool:
    """Insert a new student row.

    Returns ``True`` on success, ``False`` if the ``student_id`` already
    exists (the existing row is left untouched).
    """
    if not student_id or not name:
        logger.warning("register_student called with empty id or name")
        return False

    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO students (student_id, name) VALUES (?, ?)",
                (student_id, name),
            )
        logger.info("Registered student %s (%s)", student_id, name)
        return True
    except sqlite3.IntegrityError:
        logger.warning("Student %s already registered", student_id)
        return False


def _last_attendance_timestamp(
    conn: sqlite3.Connection, student_id: str
) -> datetime | None:
    row = conn.execute(
        "SELECT date, time FROM attendance WHERE student_id = ? "
        "ORDER BY date DESC, time DESC LIMIT 1",
        (student_id,),
    ).fetchone()
    if row is None:
        return None
    try:
        return datetime.strptime(f"{row['date']} {row['time']}", _TIMESTAMP_FORMAT)
    except ValueError:
        logger.warning("Bad timestamp for %s: %s %s", student_id, row["date"], row["time"])
        return None


def mark_attendance(
    student_id: str,
    name: str,
    confidence: float,
    session: str = "default",
) -> bool:
    """Record an attendance event for ``student_id``.

    Enforces ``config.DUPLICATE_WINDOW_SECONDS``: if the most recent
    attendance for the student is within that window, the new event is
    rejected and the function returns ``False``.

    Returns:
        ``True`` if a row was inserted, ``False`` if blocked as duplicate
        or if the student is not registered.
    """
    if not student_id:
        logger.warning("mark_attendance called with empty student_id")
        return False

    now = datetime.now()
    with _connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM students WHERE student_id = ?", (student_id,)
        ).fetchone()
        if exists is None:
            logger.warning("Attendance for unknown student %s rejected", student_id)
            return False

        last_ts = _last_attendance_timestamp(conn, student_id)
        if last_ts is not None:
            elapsed = (now - last_ts).total_seconds()
            if 0 <= elapsed < config.DUPLICATE_WINDOW_SECONDS:
                logger.warning(
                    "Duplicate attendance for %s within %.0fs window (elapsed=%.0fs)",
                    student_id,
                    config.DUPLICATE_WINDOW_SECONDS,
                    elapsed,
                )
                return False

        conn.execute(
            "INSERT INTO attendance (student_id, name, date, time, confidence, session)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                student_id,
                name,
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                float(confidence) if confidence is not None else None,
                session or "default",
            ),
        )
    logger.info("Marked attendance for %s (%s) session=%s", student_id, name, session)
    return True


def get_attendance_by_date(date: str) -> list[dict]:
    """Return all attendance records for the given ``YYYY-MM-DD`` date."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, student_id, name, date, time, confidence, session "
            "FROM attendance WHERE date = ? ORDER BY time ASC",
            (date,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_all_students() -> list[dict]:
    """Return every registered student as a list of dicts."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT student_id, name, enrolled_at FROM students ORDER BY enrolled_at"
        ).fetchall()
    return [dict(row) for row in rows]


def export_to_csv(date: str) -> Path:
    """Write the attendance for ``date`` to ``attendance_<date>.csv``.

    Returns:
        The path of the created CSV file. The file is created even when
        there are no records (only the header row is written).
    """
    config.ATTENDANCE_CSV_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.ATTENDANCE_CSV_DIR / f"attendance_{date}.csv"
    records = get_attendance_by_date(date)
    fieldnames = ["id", "student_id", "name", "date", "time", "confidence", "session"]

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({k: row.get(k) for k in fieldnames})

    logger.info("Exported %d rows to %s", len(records), out_path)
    return out_path


def delete_attendance_by_date(date: str) -> int:
    """Delete all attendance records for ``date`` (``YYYY-MM-DD``).

    Returns the number of rows removed. Enrolled students are left intact;
    only attendance entries for that day are cleared.
    """
    with _connect() as conn:
        cur = conn.execute("DELETE FROM attendance WHERE date = ?", (date,))
        removed = cur.rowcount
    logger.info("Deleted %d attendance row(s) for %s", removed, date)
    return removed


def get_attendance_summary() -> dict:
    """Return ``{student_id: {name, total_sessions, dates: []}}``.

    ``total_sessions`` is the number of distinct dates the student has
    been marked present on.
    """
    summary: dict = {}
    with _connect() as conn:
        rows = conn.execute(
            "SELECT student_id, name, date FROM attendance ORDER BY date ASC"
        ).fetchall()

    for row in rows:
        sid = row["student_id"]
        entry = summary.setdefault(
            sid, {"name": row["name"], "total_sessions": 0, "dates": []}
        )
        if row["date"] not in entry["dates"]:
            entry["dates"].append(row["date"])
        entry["name"] = row["name"]

    for entry in summary.values():
        entry["total_sessions"] = len(entry["dates"])

    return summary
