"""Wipe ALL enrolled students, attendance, encodings and face data.

This is destructive and irreversible. It clears:
  - the ``students`` and ``attendance`` tables
  - encodings.pkl
  - lbph_model.yml and lbph_labels.pkl
  - every folder under data/enrolled_faces/

Run::

    python -m api.reset_all          # asks for confirmation
    python -m api.reset_all --yes    # skip the confirmation prompt
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
import database  # noqa: E402

_LBPH_LABELS = config.PROJECT_ROOT / "data" / "encodings" / "lbph_labels.pkl"


def reset_all(verbose: bool = True) -> dict:
    """Wipe all students, attendance, encodings, models and face crops.

    Returns a small summary dict of what was removed. Safe to call from the
    API as well as the CLI.
    """
    database.init_db()

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # 1. Clear DB tables (capture counts first)
    conn = sqlite3.connect(str(config.DB_PATH))
    try:
        students = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        attendance = conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
        conn.execute("DELETE FROM attendance")
        conn.execute("DELETE FROM students")
        conn.commit()
    finally:
        conn.close()
    _log(f"Cleared {students} student(s) and {attendance} attendance row(s).")

    # 2. Remove model / encoding files
    files_removed = 0
    for f in (config.ENCODINGS_PATH, config.LBPH_MODEL_PATH, _LBPH_LABELS):
        if f.exists():
            f.unlink()
            files_removed += 1
            _log(f"Deleted {f.name}")

    # 3. Remove saved face crops
    if config.ENROLLED_FACES_DIR.exists():
        for child in config.ENROLLED_FACES_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        _log(f"Cleared {config.ENROLLED_FACES_DIR}")

    _log("\nDone. The system is back to a clean, empty state.")
    return {
        "students_removed": students,
        "attendance_removed": attendance,
        "files_removed": files_removed,
    }


if __name__ == "__main__":
    if "--yes" in sys.argv:
        reset_all()
    else:
        ans = input("This will permanently delete ALL students and data. Type 'yes' to continue: ")
        if ans.strip().lower() == "yes":
            reset_all()
        else:
            print("Aborted. Nothing was deleted.")
