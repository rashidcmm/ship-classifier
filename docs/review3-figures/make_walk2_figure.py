"""Slide 9 (walk) chart after the stricter re-run: what each feature set adds at the final set, with its noise.

  python src/feature_walk2.py --judge lr          # -> runs/feature_walk2/{steps.jsonl, result_lr.json}
  python docs/review3-figures/make_walk2_figure.py # -> docs/review3-figures/readable/09_walk_strict.png

Kept sets: final set minus final set without it. Left-out sets: final set plus the set, minus the final set; mel
mean + spread and loudness were never tried on exactly the final set, so their last retry is shown (second pass,
before MFCC came back). Bars = mean over the 5 validation folds (paired), whiskers = 2 standard errors.
Same style as make_readable_figures.py.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "readable"
INK, MUTED, GRID, SURFACE = "#10283d", "#5b7083", "#d9e1e8", "#ffffff"
BLUE, GREY, ORANGE = "#0f6fff", "#b8c4ce", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 22, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 20, "ytick.labelsize": 21,
                     "axes.labelsize": 21, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
NAMES = ["mel mean + spread", "mel percentiles", "MFCC", "1/3-octave levels", "level changes over time",
         "spectral shape", "loudness and impulsiveness", "tonal lines", "propeller modulation (DEMON)"]
SHORT = {"mel percentiles": "mel percentiles", "mel mean + spread": "mel mean + spread",
         "1/3-octave levels": "1/3-octave levels", "spectral shape": "spectral shape",
         "loudness and impulsiveness": "loudness & impulsiveness", "MFCC": "MFCC",
         "propeller modulation (DEMON)": "propeller rate (DEMON)", "tonal lines": "tonal lines",
         "level changes over time": "level changes"}
MIN_GAIN = 0.5


def canon(sets):
    return " | ".join(sorted(set(sets), key=NAMES.index))


def contributions():
    walk = ROOT / "runs" / "feature_walk2"
    rows = {}
    for line in (walk / "steps.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["judge"] == "lr":
            rows[r["groups"]] = 100 * np.array(r["val_by_fold"])
    res = json.loads((walk / "result_lr.json").read_text())
    final = res["final"].split(" | ")
    before_mfcc = [x for x in final if x != "MFCC"]          # the kept set during the second pass's first retries
    out = []
    for s in NAMES:
        if s in final:
            d = rows[canon(final)] - rows[canon([x for x in final if x != s])]
        elif canon(final + [s]) in rows:
            d = rows[canon(final + [s])] - rows[canon(final)]
        else:
            d = rows[canon(before_mfcc + [s])] - rows[canon(before_mfcc)]
        out.append((s, s in final, d.mean(), 2 * d.std(ddof=1) / np.sqrt(len(d))))
    return out


def main():
    data = sorted(contributions(), key=lambda t: (not t[1], -t[2]))[::-1]   # kept on top, largest first
    fig, ax = plt.subplots(figsize=(16, 6.9))
    y = np.arange(len(data))
    for yi, (s, kept, m, e) in zip(y, data):
        color = BLUE if kept else GREY
        ax.barh(yi, m, 0.62, color=color, zorder=2)
        ax.errorbar(m, yi, xerr=e, fmt="none", ecolor=INK, elinewidth=2, capsize=6, zorder=3)
        x = max(m + e, 0) + 0.06
        if MIN_GAIN - 0.14 < x < MIN_GAIN + 0.06:          # keep the value clear of the dashed bar line
            x = MIN_GAIN + 0.06
        ax.text(x, yi, f"{m:+.2f}", va="center", ha="left", fontsize=20, color=INK)
    ax.set_yticks(y, [f"{SHORT[s]}" + ("" if kept else "  (left out)") for s, kept, _, _ in data])
    for lab, (_, kept, _, _) in zip(ax.get_yticklabels(), data):
        lab.set_color(INK if kept else MUTED)
    ax.axvline(0, color=INK, lw=1.5, zorder=1)
    ax.axvline(MIN_GAIN, color=ORANGE, lw=2.5, ls="--", zorder=1)
    ax.text(MIN_GAIN + 0.03, len(data) - 0.45, "bar: +0.5", color=ORANGE, fontsize=20, va="bottom", weight="bold")
    ax.set_xlim(-1.25, 1.55)
    ax.set_ylim(-0.6, len(data) - 0.1)
    ax.set_xlabel("what the set adds to the final set (validation points, 6 classes + other)")
    ax.grid(axis="x", color=GRID, lw=1, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "09_walk_strict.png", dpi=100, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", OUT / "09_walk_strict.png")


if __name__ == "__main__":
    main()
