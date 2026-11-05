"""Slide 9: lesson 15's forward selection, 6 classes (no "other"), gradient boosting, test balanced accuracy per step.

  python docs/review3-figures/make_forward6_figure.py   # -> docs/review3-figures/readable/09_forward6.png
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "readable"
INK, GRID, SURFACE, BLUE = "#10283d", "#d9e1e8", "#ffffff", "#0f6fff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 22, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 20, "ytick.labelsize": 21,
                     "axes.labelsize": 21, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
SHORT = {"1/3-octave levels": "1/3-octave levels", "MFCC": "MFCC", "propeller modulation (DEMON)": "propeller rate (DEMON)",
         "level changes over time": "level changes", "tonal lines": "tonal lines",
         "loudness and impulsiveness": "loudness & impulsiveness", "spectral shape": "spectral shape"}
r = pd.DataFrame(json.loads((ROOT / "runs/six_classes_fewer/groups_results.json").read_text())["results"])
f = r[(r.kind == "forward") & (r.model == "gradient boosting")].astype({"step": int})
t = f.groupby("step").agg(added=("added", "first"), n=("n_features", "first"), acc=("balanced_accuracy", "mean"))
labels = [("start: " if s == 1 else "+ ") + SHORT[a] for s, a in zip(t.index, t.added)]
acc = 100 * t.acc.to_numpy()
fig, ax = plt.subplots(figsize=(16, 6.6))
y = np.arange(len(t))[::-1]
ax.plot(acc, y, color=BLUE, lw=3, zorder=1)
ax.scatter(acc, y, s=260, color=BLUE, zorder=3, edgecolor=SURFACE, lw=2)
for a, yy, n in zip(acc, y, t.n):
    ax.text(a + 0.18, yy + 0.05, f"{a:.1f} %", fontsize=21, color=INK, va="bottom", ha="left",
            weight="bold" if yy == 0 else "normal")
    ax.text(a + 0.18, yy - 0.08, f"{n} numbers", fontsize=15, color="#5b7083", va="top", ha="left")
ax.set_yticks(y, labels)
ax.set_xlim(53, 64)
ax.set_ylim(-0.6, len(t) - 0.4)
ax.set_xlabel("test balanced accuracy (%), 6 classes, gradient boosting (chance 16.7 %)")
ax.grid(axis="x", color=GRID, lw=1, zorder=0)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "09_forward6.png", dpi=100, bbox_inches="tight", facecolor=SURFACE)
print("wrote", OUT / "09_forward6.png")
