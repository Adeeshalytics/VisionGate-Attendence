"""Streamlit dashboard for VisionGate.

Run with::

    streamlit run dashboard.py

All data is read through :mod:`database` and the constants in
:mod:`config`. Matplotlib + seaborn are used for charts (no Plotly).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

import config
import database

PAGE_TITLE = "VisionGate — Attendance Dashboard"
ATTENDANCE_RATE_WARNING = 0.80
LOW_ATTENDANCE_BG = "#ffcccc"
ATTENDANCE_COLUMNS = ["id", "student_id", "name", "date", "time", "confidence", "session"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _attendance_df(records: list[dict]) -> pd.DataFrame:
    """Build a tidy DataFrame from attendance dict rows."""
    if not records:
        return pd.DataFrame(columns=ATTENDANCE_COLUMNS)
    df = pd.DataFrame(records, columns=ATTENDANCE_COLUMNS)
    df["confidence_pct"] = (df["confidence"].fillna(0) * 100).round(1).astype(str) + "%"
    return df


def _all_attendance_rows() -> pd.DataFrame:
    """Read every attendance row by querying each known date — keeps the
    surface small (we only depend on already-public DB functions)."""
    students = database.get_all_students()
    summary = database.get_attendance_summary()
    all_dates: set[str] = set()
    for entry in summary.values():
        all_dates.update(entry.get("dates", []))

    rows: list[dict] = []
    for d in sorted(all_dates):
        rows.extend(database.get_attendance_by_date(d))

    df = pd.DataFrame(rows, columns=ATTENDANCE_COLUMNS)
    if students:
        students_df = pd.DataFrame(students)
        df = df.merge(
            students_df[["student_id", "name"]].rename(columns={"name": "student_name"}),
            on="student_id",
            how="left",
        )
    return df


# ---------------------------------------------------------------------------
# Tab 1 — Today's Attendance
# ---------------------------------------------------------------------------

@st.fragment(run_every=30)
def render_today_tab() -> None:
    today = _today_str()
    st.subheader(f"Today — {today}")

    students = database.get_all_students()
    total_enrolled = len(students)
    today_records = database.get_attendance_by_date(today)
    present_ids = {row["student_id"] for row in today_records}
    present_count = len(present_ids)

    rate = (present_count / total_enrolled * 100.0) if total_enrolled else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Present today", present_count)
    c2.metric("Total enrolled", total_enrolled)
    c3.metric("Attendance rate", f"{rate:.1f}%")

    df = _attendance_df(today_records)
    if df.empty:
        st.info("No attendance recorded yet today.")
    else:
        display_df = df[["name", "student_id", "time", "confidence_pct"]].rename(
            columns={"confidence_pct": "confidence"}
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    if total_enrolled:
        absent = max(0, total_enrolled - present_count)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            [present_count, absent],
            labels=["Present", "Absent"],
            colors=["#2ecc71", "#e74c3c"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title("Today: Present vs Absent")
        st.pyplot(fig)
        plt.close(fig)

    st.caption("Auto-refreshes every 30 seconds.")


# ---------------------------------------------------------------------------
# Tab 2 — History
# ---------------------------------------------------------------------------

def render_history_tab() -> None:
    st.subheader("Attendance history")
    picked = st.date_input("Pick a date", value=date.today(), key="history_date")
    picked_str = picked.strftime("%Y-%m-%d")

    records = database.get_attendance_by_date(picked_str)
    df = _attendance_df(records)

    if df.empty:
        st.info(f"No attendance recorded on {picked_str}.")
    else:
        display_df = df[["student_id", "name", "time", "confidence_pct", "session"]].rename(
            columns={"confidence_pct": "confidence"}
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv_path: Path = database.export_to_csv(picked_str)
    try:
        csv_bytes = csv_path.read_bytes()
        st.download_button(
            label=f"Download CSV ({csv_path.name})",
            data=csv_bytes,
            file_name=csv_path.name,
            mime="text/csv",
        )
    except OSError as exc:
        st.error(f"Could not read CSV file: {exc}")


# ---------------------------------------------------------------------------
# Tab 3 — Student Registry
# ---------------------------------------------------------------------------

def render_registry_tab() -> None:
    st.subheader("Enrolled students")
    students = database.get_all_students()
    summary = database.get_attendance_summary()

    if not students:
        st.info("No students enrolled yet.")
        return

    rows = []
    for s in students:
        sid = s["student_id"]
        attended = summary.get(sid, {}).get("total_sessions", 0)
        rows.append(
            {
                "student_id": sid,
                "name": s["name"],
                "enrolled_at": s.get("enrolled_at"),
                "total_sessions": attended,
            }
        )
    df = pd.DataFrame(rows)

    def _highlight_zero(row):
        if row["total_sessions"] == 0:
            return [f"background-color: {LOW_ATTENDANCE_BG}"] * len(row)
        return [""] * len(row)

    styled = df.style.apply(_highlight_zero, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 4 — Analytics
# ---------------------------------------------------------------------------

def render_analytics_tab() -> None:
    st.subheader("Top attenders")
    summary = database.get_attendance_summary()
    students = database.get_all_students()
    student_name = {s["student_id"]: s["name"] for s in students}

    if not summary:
        st.info("No attendance data available yet.")
        return

    leaderboard = (
        pd.DataFrame(
            [
                {
                    "student_id": sid,
                    "name": info.get("name") or student_name.get(sid, sid),
                    "sessions_attended": info.get("total_sessions", 0),
                }
                for sid, info in summary.items()
            ]
        )
        .sort_values("sessions_attended", ascending=False)
        .head(10)
    )

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=leaderboard,
        y="name",
        x="sessions_attended",
        ax=ax1,
        palette="viridis",
        hue="name",
        legend=False,
    )
    ax1.set_xlabel("Sessions attended")
    ax1.set_ylabel("")
    ax1.set_title("Top 10 students by attendance count")
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    st.subheader("Daily attendance — last 14 days")
    days = [date.today() - timedelta(days=i) for i in range(13, -1, -1)]
    counts = [len(database.get_attendance_by_date(d.strftime("%Y-%m-%d"))) for d in days]
    daily_df = pd.DataFrame({"date": days, "count": counts})

    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    ax2.plot(daily_df["date"], daily_df["count"], marker="o", color="#3498db")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Records")
    ax2.set_title("Attendance records per day")
    fig2.autofmt_xdate(rotation=30)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.subheader("Students below 80% attendance")
    all_dates: set[str] = set()
    for info in summary.values():
        all_dates.update(info.get("dates", []))
    total_sessions_held = len(all_dates)

    if total_sessions_held == 0:
        st.info("Not enough data to compute attendance rates.")
        return

    rate_rows = []
    for s in students:
        sid = s["student_id"]
        attended = summary.get(sid, {}).get("total_sessions", 0)
        rate = attended / total_sessions_held
        rate_rows.append(
            {
                "student_id": sid,
                "name": s["name"],
                "sessions_attended": attended,
                "sessions_held": total_sessions_held,
                "attendance_rate": rate,
            }
        )
    rate_df = pd.DataFrame(rate_rows).sort_values("attendance_rate")

    def _highlight_low(row):
        if row["attendance_rate"] < ATTENDANCE_RATE_WARNING:
            return [f"background-color: {LOW_ATTENDANCE_BG}"] * len(row)
        return [""] * len(row)

    display_df = rate_df.copy()
    display_df["attendance_rate"] = (display_df["attendance_rate"] * 100).round(1).astype(
        str
    ) + "%"
    styled = display_df.style.apply(
        lambda r: _highlight_low(rate_df.loc[r.name]), axis=1
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="📷", layout="wide")
    st.title(PAGE_TITLE)
    st.caption(f"Database: {config.DB_PATH}")

    database.init_db()
    sns.set_theme(style="whitegrid")

    tab_today, tab_history, tab_registry, tab_analytics = st.tabs(
        ["Today's Attendance", "History", "Student Registry", "Analytics"]
    )
    with tab_today:
        render_today_tab()
    with tab_history:
        render_history_tab()
    with tab_registry:
        render_registry_tab()
    with tab_analytics:
        render_analytics_tab()


if __name__ == "__main__":
    main()
