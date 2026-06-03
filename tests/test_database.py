"""Tests for the database module."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

import config
import database


@pytest.fixture
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``config.DB_PATH`` and the CSV dir at a per-test tmp directory."""
    db_file = tmp_path / "attendance.db"
    csv_dir = tmp_path / "csv"
    monkeypatch.setattr(config, "DB_PATH", db_file)
    monkeypatch.setattr(config, "ATTENDANCE_CSV_DIR", csv_dir)
    monkeypatch.setattr(config, "DUPLICATE_WINDOW_SECONDS", 300)
    database.init_db()
    return db_file


def test_init_db_creates_tables(tmp_db: Path) -> None:
    import sqlite3

    conn = sqlite3.connect(str(tmp_db))
    try:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    finally:
        conn.close()
    assert {"students", "attendance"}.issubset(names)


def test_register_student_inserts_correctly(tmp_db: Path) -> None:
    assert database.register_student("EG2020001", "Alice") is True
    students = database.get_all_students()
    assert len(students) == 1
    assert students[0]["student_id"] == "EG2020001"
    assert students[0]["name"] == "Alice"


def test_duplicate_student_id_is_handled(tmp_db: Path) -> None:
    assert database.register_student("EG2020002", "Bob") is True
    assert database.register_student("EG2020002", "Bobby") is False
    students = database.get_all_students()
    assert len(students) == 1
    assert students[0]["name"] == "Bob"  # original row preserved


def test_mark_attendance_inserts_record(tmp_db: Path) -> None:
    database.register_student("EG2020003", "Carol")
    assert database.mark_attendance("EG2020003", "Carol", 0.91, "session_A") is True

    today = datetime.now().strftime("%Y-%m-%d")
    rows = database.get_attendance_by_date(today)
    assert len(rows) == 1
    assert rows[0]["student_id"] == "EG2020003"
    assert rows[0]["name"] == "Carol"
    assert rows[0]["session"] == "session_A"
    assert rows[0]["confidence"] == pytest.approx(0.91)


def test_duplicate_attendance_within_window_is_blocked(tmp_db: Path) -> None:
    database.register_student("EG2020004", "Dave")
    assert database.mark_attendance("EG2020004", "Dave", 0.88) is True
    # Immediately retry — must be blocked because elapsed < 300s.
    assert database.mark_attendance("EG2020004", "Dave", 0.88) is False

    today = datetime.now().strftime("%Y-%m-%d")
    assert len(database.get_attendance_by_date(today)) == 1


def test_export_to_csv_creates_file(tmp_db: Path) -> None:
    database.register_student("EG2020005", "Eve")
    database.mark_attendance("EG2020005", "Eve", 0.77, "morning")

    today = datetime.now().strftime("%Y-%m-%d")
    out_path = database.export_to_csv(today)

    assert out_path.exists()
    assert out_path.name == f"attendance_{today}.csv"

    content = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert content[0].split(",") == [
        "id",
        "student_id",
        "name",
        "date",
        "time",
        "confidence",
        "session",
    ]
    assert len(content) >= 2
    assert "EG2020005" in content[1]
    assert "Eve" in content[1]
