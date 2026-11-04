"""Slide figures for lessons 22-24 ("other" always a class), from runs/with_other/.

  python docs/review3-figures/make_with_other_figures.py      # -> docs/review3-figures/2*_*.png

Static PNGs for Canva (1600 x 900, light). Two series colours (deck blue = gradient boosting, orange = logistic
regression); every bar carries its value, so colour is never the only cue.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import with_other as W

OUT = Path(__file__).resolve().parent
INK, MUTED, GRID, SURFACE = "#10283d", "#5b7083", "#d9e1e8", "#ffffff"
GB, LR, GREY = "#0f6fff", "#eb6834", "#b8c4ce"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def log(name):
    return pd.DataFrame(W.Log(W.OUT_DIR / f"{name}.jsonl").rows)


def save(fig, name):
    fig.savefig(OUT / name, dpi=100, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / name)


def bars(ax, labels, values, color, offset, width, fmt="{:.1f}"):
    y = np.arange(len(labels)) + offset
    ax.barh(y, values, width, color=color, edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, values):
        ax.text(v + 0.4, yi, fmt.format(v), va="center", fontsize=13, color=INK)


def fig_sets():
    m = log("main")
    size = [257, 641, 144, 401, 785]                     # lesson 22's order of the five sets
    order = sorted(set(m.features), key=lambda f: size.index(int(f.rsplit("(", 1)[1].rstrip(")"))))
    fig, ax = plt.subplots(figsize=(16, 7.5))
    for model, color, off in (("gradient boosting", GB, -0.2), ("logistic regression", LR, 0.2)):
        g = m[m.model == model].groupby("features").balanced_accuracy.mean() * 100
        bars(ax, order, [g[s] for s in order], color, off, 0.38)
    ax.set_yticks(range(len(order)), [s.replace(" + ", " + ") for s in order], fontsize=14, color=INK)
    ax.invert_yaxis(); ax.set_xlim(0, 70); ax.axvline(100 / 7, color=MUTED, ls="--", lw=1)
    ax.text(100 / 7 + 0.4, len(order) - 0.45, "chance 14.3 %", color=MUTED, fontsize=12)
    ax.set_xlabel("balanced accuracy per clip (%), mean of 5 test folds")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GB), plt.Rectangle((0, 0), 1, 1, color=LR)],
              labels=["gradient boosting", "logistic regression"], loc="lower right", frameon=False, fontsize=13)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    save(fig, "22_feature_sets_with_other.png")


def fig_walk():
    w = log("walk")
    res = json.loads((W.OUT_DIR / "walk_result.json").read_text())
    first = res["order"][0]
    alone = w[(w.stage == "alone") & (w.groups == first)].iloc[0]
    steps = w[w.stage == "walk"].sort_values("step")
    labels = [f"1. {first}"] + [f"{int(s.step)}. + {s.added}" for s in steps.itertuples()]
    vals = [100 * alone.val] + [100 * s.val for s in steps.itertuples()]
    kept = [True] + [s.val > s.before for s in steps.itertuples()]
    ms = [alone.ms] + [s.ms for s in steps.itertuples()]
    nf = [alone.n_features] + [s.n_features for s in steps.itertuples()]
    fig, ax = plt.subplots(figsize=(16, 8))
    y = np.arange(len(labels))
    # dots, not bars: the axis is zoomed to show 1-point steps, and bars would exaggerate them
    ky = [yi for yi, k in zip(y, kept) if k]
    ax.plot([v for v, k in zip(vals, kept) if k], ky, color=GB, lw=2, zorder=2)
    ax.scatter([v for v, k in zip(vals, kept) if k], ky, s=140, color=GB, zorder=3, label="kept (validation went up)")
    ax.scatter([v for v, k in zip(vals, kept) if not k], [yi for yi, k in zip(y, kept) if not k], s=140,
               facecolor=SURFACE, edgecolor=MUTED, linewidth=2, zorder=3, label="left out (no gain)")
    for yi, v, k, c, n in zip(y, vals, kept, ms, nf):
        ax.text(54.2, yi, f"{v:.1f} %  {'kept' if k else 'left out'}  ·  {n} numbers, {c:.0f} ms/clip",
                va="center", fontsize=12.5, color=INK if k else MUTED)
    ax.legend(loc="lower left", frameon=False, fontsize=12)
    ax.set_yticks(y, labels, fontsize=13.5, color=INK); ax.invert_yaxis()
    ax.set_xlim(45, 62); ax.set_xticks(range(46, 54, 2))
    ax.set_xlabel("validation balanced accuracy (%), mean of 5 rounds (logistic regression)", x=0.32)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    save(fig, "23_walk_one_at_a_time.png")


def fig_bands():
    b = log("bands")
    if b.empty:
        return
    order = list(dict.fromkeys(b.features))
    fig, ax = plt.subplots(figsize=(16, 7))
    for model, color, off in (("gradient boosting", GB, -0.2), ("logistic regression", LR, 0.2)):
        g = b[b.model == model].groupby("features").balanced_accuracy.mean() * 100
        bars(ax, order, [g.get(s, np.nan) for s in order], color, off, 0.38)
    ax.set_yticks(range(len(order)), order, fontsize=14, color=INK); ax.invert_yaxis(); ax.set_xlim(0, 62)
    ax.set_xlabel("balanced accuracy per clip (%), mel band mean + spread only, 6 classes + other")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GB), plt.Rectangle((0, 0), 1, 1, color=LR)],
              labels=["gradient boosting", "logistic regression"], loc="lower right", frameon=False, fontsize=13)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    save(fig, "23_mel_bands_with_other.png")


def fig_merged():
    m = log("merged")
    if m.empty:
        return
    m = m[~m.features.str.startswith("PANN")]
    without = {"6 classes + other": (61.9, 70.5), "5 classes + other": (69.9, 78.9), "4 classes + other": (80.2, 86.8)}
    gb = m[m.model == "gradient boosting"].groupby("grouping")[["balanced_accuracy", "per_ship_balanced_accuracy"]].mean() * 100
    order = [g for g in W.GROUPINGS if g in gb.index]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
    for ax, col, i, title in ((axes[0], "balanced_accuracy", 0, "per clip"), (axes[1], "per_ship_balanced_accuracy", 1, "per ship")):
        x = np.arange(len(order))
        old = [without[g][i] for g in order]
        new = [gb.loc[g, col] for g in order]
        ax.bar(x - 0.2, old, 0.38, color=GREY, edgecolor=SURFACE, linewidth=2, label="without 'other' (lessons 13-20, 144 engineered)")
        ax.bar(x + 0.2, new, 0.38, color=GB, edgecolor=SURFACE, linewidth=2, label="with 'other' (lesson 24, engineered + mel, 401)")
        for xi, o, n in zip(x, old, new):
            ax.text(xi - 0.2, o + 1, f"{o:.1f}", ha="center", fontsize=13, color=MUTED)
            ax.text(xi + 0.2, n + 1, f"{n:.1f}", ha="center", fontsize=13, color=INK)
        ax.set_xticks(x, [g.replace(" + other", "") for g in order], color=INK); ax.set_ylim(0, 100)
        ax.axhline(75, color=MUTED, ls="--", lw=1); ax.text(-0.45, 76, "goal 75 %", color=MUTED, fontsize=12, ha="left")
        ax.set_title(title, color=INK); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    axes[0].set_ylabel("balanced accuracy (%), gradient boosting, 5 test folds")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=13, bbox_to_anchor=(0.5, -0.06))
    save(fig, "24_merged_with_other.png")


if __name__ == "__main__":
    fig_sets()
    fig_walk()
    fig_bands()
    fig_merged()
