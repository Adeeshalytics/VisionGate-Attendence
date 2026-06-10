"""Render the system architecture diagram as a PNG for the slides.

    python make_architecture_diagram.py   ->  data/validation/architecture.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import config

OUT = config.PROJECT_ROOT / "data" / "validation" / "architecture.png"

# Colour palette (slide-friendly, dark-on-light)
C_INPUT = "#dbeafe"      # light blue
C_PRE = "#e9d5ff"        # light violet
C_DETECT = "#fde68a"     # amber
C_ENROLL = "#bbf7d0"     # green
C_RECOG = "#bae6fd"      # sky
C_STORE = "#fecaca"      # red/pink
C_DB = "#fed7aa"         # orange
C_UI = "#c7d2fe"         # indigo
EDGE = "#334155"


def _box(ax, xy, w, h, text, color, fontsize=10, bold=False):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4, edgecolor=EDGE, facecolor=color, zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        fontweight="bold" if bold else "normal", color="#0f172a", zorder=3,
        wrap=True,
    )
    return (x, y, w, h)


def _arrow(ax, p1, p2, color=EDGE, style="-|>", lw=1.6, ls="-"):
    arr = FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color, linestyle=ls, zorder=1,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arr)


def bottom(b):
    x, y, w, h = b
    return (x + w / 2, y)


def top(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


def left(b):
    x, y, w, h = b
    return (x, y + h / 2)


def right(b):
    x, y, w, h = b
    return (x + w, y + h / 2)


def main() -> None:
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.axis("off")

    ax.text(6, 11.6, "VisionGate — System Architecture",
            ha="center", fontsize=16, fontweight="bold", color="#0f172a")
    ax.text(6, 11.15, "Face-Recognition Attendance · University of Ruhuna · EE7204/EC7205",
            ha="center", fontsize=9.5, color="#475569")

    # --- shared top pipeline ---
    cam = _box(ax, (4.7, 10.0), 2.6, 0.8, "Webcam capture", C_INPUT, 11, True)
    pre = _box(ax, (4.2, 8.7), 3.6, 0.9,
               "Preprocessing\nresize · denoise · CLAHE", C_PRE, 10)
    det = _box(ax, (4.1, 7.3), 3.8, 0.9,
               "Face Detection\nMediaPipe BlazeFace → Haar fallback", C_DETECT, 9.5)

    _arrow(ax, bottom(cam), top(pre))
    _arrow(ax, bottom(pre), top(det))

    # --- branch: enrollment (left) vs recognition (right) ---
    # Enrollment column
    en1 = _box(ax, (0.4, 5.7), 3.5, 0.85,
               "Crop ROI + Augment\n(7 variants / face)", C_ENROLL, 9.5)
    en2 = _box(ax, (0.4, 4.4), 3.5, 0.85,
               "dlib ResNet 128-d encode\n+ train LBPH baseline", C_ENROLL, 9.5)
    en3 = _box(ax, (0.4, 3.1), 3.5, 0.85,
               "Save models\nencodings.pkl · lbph_model.yml", C_STORE, 9.5)
    _arrow(ax, bottom(en1), top(en2))
    _arrow(ax, bottom(en2), top(en3))

    # Recognition column
    rc1 = _box(ax, (8.1, 5.7), 3.5, 0.85, "Crop ROI", C_RECOG, 9.5)
    rc2 = _box(ax, (8.1, 4.65), 3.5, 0.7, "dlib 128-d encode", C_RECOG, 9.5)
    rc3 = _box(ax, (8.1, 3.6), 3.5, 0.7,
               "Match: distance ≤ 0.45", C_RECOG, 9.5)
    rc4 = _box(ax, (8.1, 2.55), 3.5, 0.7,
               "Liveness: EAR blink", C_RECOG, 9.5)
    _arrow(ax, bottom(rc1), top(rc2))
    _arrow(ax, bottom(rc2), top(rc3))
    _arrow(ax, bottom(rc3), top(rc4))

    # detection feeds both branches
    _arrow(ax, bottom(det), top(en1), ls="-")
    _arrow(ax, bottom(det), top(rc1), ls="-")

    # branch labels
    ax.text(2.15, 6.75, "ENROLLMENT", ha="center", fontsize=10,
            fontweight="bold", color="#15803d")
    ax.text(9.85, 6.75, "RECOGNITION", ha="center", fontsize=10,
            fontweight="bold", color="#0369a1")

    # saved models feed the matcher (dashed)
    _arrow(ax, right(en3), (8.1, rc3[1] + rc3[3] / 2),
           color="#b91c1c", ls="--", lw=1.4)
    ax.text(6.0, 3.55, "gallery", ha="center", fontsize=8, color="#b91c1c")

    # --- database ---
    db = _box(ax, (4.6, 1.2), 2.8, 0.85,
              "SQLite DB\nstudents · attendance", C_DB, 10, True)
    _arrow(ax, bottom(rc4), (db[0] + db[2], db[1] + db[3] / 2))
    # enrollment registers students too
    _arrow(ax, bottom(en3), (db[0], db[1] + db[3] / 2), ls="-")

    # --- outputs / UIs ---
    ui = _box(ax, (3.0, 0.05), 6.0, 0.75,
              "Web dashboard: FastAPI + Next.js  |  CSV export  |  Validation suite",
              C_UI, 9)
    _arrow(ax, bottom(db), top(ui))

    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Architecture diagram saved to {OUT}")


if __name__ == "__main__":
    main()
