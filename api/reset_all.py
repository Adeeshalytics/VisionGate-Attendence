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


def reset_all() -> None:
    database.init_db()

    # 1. Clear DB tables
    conn = sqlite3.connect(str(config.DB_PATH))
    try:
        conn.execute("DELETE FROM attendance")
        conn.execute("DELETE FROM students")
        conn.commit()
    finally:
        conn.close()
    print("Cleared students and attendance tables.")

    # 2. Remove model / encoding files
    for f in (config.ENCODINGS_PATH, config.LBPH_MODEL_PATH, _LBPH_LABELS):
        if f.exists():
            f.unlink()
            print(f"Deleted {f.name}")

    # 3. Remove saved face crops
    if config.ENROLLED_FACES_DIR.exists():
        for child in config.ENROLLED_FACES_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        print(f"Cleared {config.ENROLLED_FACES_DIR}")

    print("\nDone. The system is back to a clean, empty state.")


if __name__ == "__main__":
    if "--yes" in sys.argv:
        reset_all()
    else:
        ans = input("This will permanently delete ALL students and data. Type 'yes' to continue: ")
        if ans.strip().lower() == "yes":
            reset_all()
        else:
            print("Aborted. Nothing was deleted.")
