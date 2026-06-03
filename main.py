"""VisionGate CLI entry point.

Run with::

    python main.py

Presents a numbered menu that dispatches into enrollment, the recognition
loop, the Streamlit dashboard, CSV export and a summary view.
"""

from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import config
import database
from utils import ensure_dirs, get_logger

logger = get_logger("main")

DASHBOARD_PORT = 8501

MENU_BANNER = """
╔══════════════════════════════════════════╗
║   VisionGate — Attendance System         ║
║   University of Ruhuna | EE7204/EC7205   ║
╚══════════════════════════════════════════╝
""".strip()

MENU_OPTIONS = """
  1. Enroll New Student
  2. Start Attendance Session
  3. Open Dashboard  (opens browser to localhost:8501)
  4. Export Today's Attendance to CSV
  5. View Today's Summary (console table)
  6. Exit
""".rstrip()


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

def _action_enroll() -> None:
    from enroll import run_enrollment

    try:
        run_enrollment()
    except SystemExit as exc:
        logger.info("Enrollment exited: %s", exc)
    except KeyboardInterrupt:
        logger.warning("Enrollment interrupted by user")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Enrollment failed: %s", exc)
        print(f"Enrollment failed: {exc}")


def _action_attendance() -> None:
    from recognize import run_recognition

    try:
        run_recognition()
    except KeyboardInterrupt:
        logger.warning("Recognition interrupted by user")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Recognition failed: %s", exc)
        print(f"Recognition failed: {exc}")


def _silence_streamlit_welcome() -> None:
    """Pre-create ``~/.streamlit/credentials.toml`` so Streamlit's first-run
    email prompt never appears. Without this, the prompt is read from the
    parent terminal's stdin and silently steals our menu input."""
    credentials_path = Path.home() / ".streamlit" / "credentials.toml"
    if credentials_path.exists():
        return
    try:
        credentials_path.parent.mkdir(parents=True, exist_ok=True)
        credentials_path.write_text('[general]\nemail = ""\n', encoding="utf-8")
        logger.info("Wrote %s to skip Streamlit welcome prompt", credentials_path)
    except OSError as exc:
        logger.warning("Could not write Streamlit credentials file: %s", exc)


def _action_dashboard() -> None:
    dashboard_path = config.PROJECT_ROOT / "dashboard.py"
    if not dashboard_path.exists():
        print(f"Dashboard script not found at {dashboard_path}")
        return

    _silence_streamlit_welcome()

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.port",
        str(DASHBOARD_PORT),
    ]
    logger.info("Launching dashboard: %s", " ".join(cmd))
    try:
        # stdin=DEVNULL detaches the child so it cannot consume our menu input.
        subprocess.Popen(
            cmd,
            cwd=str(config.PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        print(f"Could not launch Streamlit: {exc}")
        return

    url = f"http://localhost:{DASHBOARD_PORT}"
    print(f"Dashboard launching at {url}")
    try:
        webbrowser.open(url, new=2)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Browser open failed: %s", exc)


def _action_export_today() -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        path: Path = database.export_to_csv(today)
        print(f"Today's attendance exported to: {path}")
    except Exception as exc:  # noqa: BLE001
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


# ---------------------------------------------------------------------------
# Menu loop
# ---------------------------------------------------------------------------

_ACTIONS = {
    "1": ("Enroll New Student", _action_enroll),
    "2": ("Start Attendance Session", _action_attendance),
    "3": ("Open Dashboard", _action_dashboard),
    "4": ("Export Today's Attendance to CSV", _action_export_today),
    "5": ("View Today's Summary", _action_view_summary),
}


def _restore_stdin_if_closed() -> None:
    """Re-open ``sys.stdin`` if a library closed it under our feet.

    Some third-party libraries (notably ``face_recognition``) call the
    builtin ``quit()`` on import-time failures, which closes the Python
    stdin wrapper but leaves file descriptor 0 open. Without this guard
    the next ``input()`` in the menu would raise
    ``ValueError: I/O operation on closed file.``
    """
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
            choice = input("Choice [1-6]: ").strip()
        except (EOFError, ValueError):
            return "6"
        if choice in _ACTIONS or choice == "6":
            return choice
        print("Invalid choice — please enter a number from 1 to 6.")


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
        if choice == "6":
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
