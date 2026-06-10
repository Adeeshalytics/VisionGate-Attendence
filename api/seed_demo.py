"""Insert demo data for frontend development / presentation rehearsal.

    python -m api.seed_demo          # insert demo students + attendance
    python -m api.seed_demo --clear  # remove ONLY the demo rows

Demo IDs use a DEMO_ prefix so --clear can drop them without touching real
enrollments.
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import database

DEMO_PREFIX = "DEMO_"
DEMO_STUDENTS = [
    (f"{DEMO_PREFIX}EG001", "Ada Lovelace"),
    (f"{DEMO_PREFIX}EG002", "Alan Turing"),
    (f"{DEMO_PREFIX}EG003", "Grace Hopper"),
    (f"{DEMO_PREFIX}EG004", "Linus Torvalds"),
    (f"{DEMO_PREFIX}EG005", "Margaret Hamilton"),
]


def _clear() -> None:
    database.init_db()
    conn = sqlite3.connect(str(config.DB_PATH))
    try:
        conn.execute(
            "DELETE FROM attendance WHERE student_id LIKE ?", (f"{DEMO_PREFIX}%",)
        )
        conn.execute(
            "DELETE FROM students WHERE student_id LIKE ?", (f"{DEMO_PREFIX}%",)
        )
        conn.commit()
    finally:
        conn.close()
    print("Demo rows removed.")


def _seed() -> None:
    database.init_db()
    for sid, name in DEMO_STUDENTS:
        database.register_student(sid, name)

    # Build attendance across the last several days with varied presence so
    # the analytics views have something interesting to show.
    conn = sqlite3.connect(str(config.DB_PATH))
    try:
        attendance_plan = {
            0: DEMO_STUDENTS,                 # today: everyone
            1: DEMO_STUDENTS[:4],
            2: DEMO_STUDENTS[:3],
            3: DEMO_STUDENTS[:5],
            5: DEMO_STUDENTS[1:4],
            7: DEMO_STUDENTS[:2],
        }
        for days_ago, present in attendance_plan.items():
            day = datetime.now() - timedelta(days=days_ago)
            for i, (sid, name) in enumerate(present):
                ts = day.replace(hour=9, minute=0, second=0) + timedelta(minutes=i * 2)
                conn.execute(
                    "INSERT INTO attendance (student_id, name, date, time, confidence, session)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        sid,
                        name,
                        ts.strftime("%Y-%m-%d"),
                        ts.strftime("%H:%M:%S"),
                        round(0.80 + 0.03 * i, 2),
                        ts.strftime("%Y-%m-%d_%H-00"),
                    ),
                )
        conn.commit()
    finally:
        conn.close()
    print(f"Seeded {len(DEMO_STUDENTS)} demo students with attendance history.")


if __name__ == "__main__":
    if "--clear" in sys.argv:
        _clear()
    else:
        _seed()
