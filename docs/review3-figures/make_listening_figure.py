"""Slide charts for "per clip, per encounter, per ship" and listening time (lesson 13's model, 6 classes).

  python src/listening_time.py                              # -> runs/listening_time/summary.csv
  python docs/review3-figures/make_listening_figure.py      # -> docs/review3-figures/readable/10_*.png

Same rules as make_readable_figures.py: one message per chart, big type, values on the marks, no chart titles.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "readable"
INK, MUTED, GRID, SURFACE = "#10283d", "#5b7083", "#d9e1e8", "#ffffff"
BLUE, LIGHT = "#0f6fff", "#9cc2ff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 22, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 21, "ytick.labelsize": 21,
                     "axes.labelsize": 22, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
SOURCE = "6 classes (lesson 13)"


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=100, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / name)


def main():
    s = pd.read_csv(ROOT / "runs" / "listening_time" / "summary.csv")
    s = s[s.source == SOURCE].set_index("n_clips")

    # 1. one answer per clip, per passage (one encounter), per ship (all its passages)
    vals = [100 * s.loc["1", "balanced_all"], 100 * s.loc["whole passage", "balanced_all"],
            100 * s.loc["whole ship", "balanced_all"]]
    labels = ["one 5-s clip", "one passage\n(one encounter)", "one ship\n(all its passages)"]
    fig, ax = plt.subplots(figsize=(11, 7))
    x = np.arange(3)
    ax.bar(x, vals, 0.6, color=[BLUE, BLUE, LIGHT])
    for xi, v in zip(x, vals):
        ax.text(xi, v + 1.2, f"{v:.1f} %", ha="center", va="bottom", fontsize=28, color=INK, weight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 85); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.axhline(100 / 6, color=MUTED, ls="--", lw=1.5)
    ax.text(2.45, 100 / 6 + 1, "chance 16.7 %", ha="right", va="bottom", color=MUTED, fontsize=19)
    ax.set_ylabel("balanced accuracy, 6 classes")
    save(fig, "10_clip_passage_ship.png")

    # 2. listening longer to the same passage: the four classes with long passages
    n = [1, 2, 3, 4, 6, 9]
    acc = [100 * s.loc[str(k), "balanced_long4"] for k in n]
    span = [s.loc[str(k), "median_span_s"] for k in n]
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot([5 * k for k in n], acc, color=BLUE, lw=3, marker="o", ms=13, mec=SURFACE, mew=2)
    for k, a in zip(n, acc):
        ax.text(5 * k, a + 0.6, f"{a:.1f} %", ha="center", va="bottom", fontsize=22, color=INK)
    ax.set_xticks([5 * k for k in n], [f"{5 * k} s" for k in n])
    ax.set_xlabel("audio heard from one passage (5-s clips averaged)")
    ax.set_ylabel("balanced accuracy (%)")
    ax.set_ylim(50, 65)
    ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
    ax.text(0.99, 0.04, f"fishing, pleasure craft, tanker, tug · 45 s of clips span about {span[-1] / 60:.0f} min",
            transform=ax.transAxes, ha="right", fontsize=18, color=MUTED)
    save(fig, "10_listening_time.png")


if __name__ == "__main__":
    main()
