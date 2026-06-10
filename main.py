"""Terminal menu for VisionGate — run `python main.py`."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import database
from utils import ensure_dirs, get_logger

logger = get_logger("main")

MENU_BANNER = """
╔══════════════════════════════════════════╗
║   VisionGate — Attendance System         ║
║   University of Ruhuna | EE7204/EC7205   ║
╚══════════════════════════════════════════╝
""".strip()

MENU_OPTIONS = """
  1. Enroll New Student
  2. Start Attendance Session
  3. Export Today's Attendance to CSV
  4. View Today's Summary
  5. Exit
""".rstrip()


def _action_enroll() -> None:
    from enroll import run_enrollment

    try:
        run_enrollment()
    except SystemExit as exc:
        logger.info("Enrollment exited: %s", exc)
    except KeyboardInterrupt:
        logger.warning("Enrollment interrupted by user")
    except Exception as exc:
        logger.exception("Enrollment failed: %s", exc)
        print(f"Enrollment failed: {exc}")


def _action_attendance() -> None:
    from recognize import run_recognition

    try:
        run_recognition()
    except KeyboardInterrupt:
        logger.warning("Recognition interrupted by user")
    except Exception as exc:
        logger.exception("Recognition failed: %s", exc)
        print(f"Recognition failed: {exc}")


def _action_export_today() -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        path: Path = database.export_to_csv(today)
        print(f"Today's attendance exported to: {path}")
    except Exception as exc:
        logger.exception("CSV export failed: %s", exc)
        print(f"CSV export failed: {exc}")


def _action_view_summary() -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    rows = database.get_attendance_by_date(today)
    if not rows:
        print(f"No attendance recorded for {today}.")
        return

    headers = ["student_id", "name", "time", "confidence", "session"]
    widths = {h: max(len(h), max(len(str(r.get(h, "") or "")) for r in rows)) for h in headers}

    def _fmt_row(values: list[str]) -> str:
        return "  ".join(values[i].ljust(widths[headers[i]]) for i in range(len(headers)))

    print()
    print(f"Today's attendance — {today}")
    print(_fmt_row(headers))
    print(_fmt_row(["-" * widths[h] for h in headers]))
    for row in rows:
        conf = row.get("confidence")
        conf_str = f"{conf * 100:.1f}%" if conf is not None else ""
        print(
            _fmt_row(
                [
                    str(row.get("student_id", "")),
                    str(row.get("name", "")),
                    str(row.get("time", "")),
                    conf_str,
                    str(row.get("session", "")),
                ]
            )
        )
    print(f"\nTotal records: {len(rows)}")


_ACTIONS = {
    "1": ("Enroll New Student", _action_enroll),
    "2": ("Start Attendance Session", _action_attendance),
    "3": ("Export Today's Attendance to CSV", _action_export_today),
    "4": ("View Today's Summary", _action_view_summary),
}


def _restore_stdin_if_closed() -> None:
    # face_recognition calls quit() if its model package fails to import, which
    # closes sys.stdin but leaves fd 0 open. Re-wrap it so the next input() works.
    stdin = sys.stdin
    if stdin is None or getattr(stdin, "closed", False):
        try:
            sys.stdin = os.fdopen(0, "r", closefd=False)
            logger.info("Restored sys.stdin after it was closed by a library")
        except OSError as exc:
            logger.error("Could not restore stdin: %s", exc)


def _prompt_choice() -> str:
    while True:
        _restore_stdin_if_closed()
        try:
            choice = input("Choice [1-5]: ").strip()
        except (EOFError, ValueError):
            return "5"
        if choice in _ACTIONS or choice == "5":
            return choice
        print("Invalid choice — please enter a number from 1 to 5.")


def main() -> None:
    ensure_dirs()
    database.init_db()
    logger.info("VisionGate started")

    while True:
        print()
        print(MENU_BANNER)
        print(MENU_OPTIONS)
        print()

        choice = _prompt_choice()
        if choice == "5":
            print("Goodbye.")
            logger.info("VisionGate exited via menu")
            return

        label, action = _ACTIONS[choice]
        logger.info("Menu selection: %s — %s", choice, label)
        try:
            action()
        except KeyboardInterrupt:
            print("\nInterrupted — returning to menu.")
            logger.warning("Action interrupted by user")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye.")
