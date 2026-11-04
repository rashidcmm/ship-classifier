"""Slide figure for lesson 26 (unknown ships with the CNN, per ship), from runs/unknown_cnn/.

  python docs/review3-figures/make_unknown_cnn_figure.py      # -> docs/review3-figures/26_unknown_cnn_per_ship.png

Static PNG for Canva, light, same palette as make_with_other_figures.py. Left: each ship's confidence (known test
ships against never-heard ships), with the threshold that keeps 95 % of known validation ships. Right: share of the
never-heard ships called "unknown", pooled over the five folds, with ship counts on every bar.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "unknown_cnn"
OUT = Path(__file__).resolve().parent
INK, MUTED, GRID, SURFACE = "#10283d", "#5b7083", "#d9e1e8", "#ffffff"
KNOWN, UNKNOWN = "#0f6fff", "#b03a2e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
TYPES = {"sailing": "sailing", "passengership": "passenger ship", "other": "other (dataset)", "rescue": "rescue",
         "dredger": "dredger", "pilotvessel": "pilot vessel"}


def ships():
    res = pd.DataFrame(json.loads((RUN / "results.json").read_text())["results"]).set_index("fold")
    p = pd.read_parquet(RUN / "predictions.parquet")
    cols = [c for c in p.columns if c.startswith("p_")]
    s = (p[p.split.isin(["test", "unknown"])].groupby(["fold", "split", "ship"])
         .agg(**{c: (c, "mean") for c in cols}, label=("label", "first")).reset_index())
    s["conf"] = s[cols].max(axis=1)
    s["caught"] = s.conf < s.fold.map(res.threshold_ship)
    return s, res


def main():
    s, res = ships()
    known, unknown = s[s.split == "test"], s[s.split == "unknown"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw={"width_ratios": [1.15, 1]})

    x = np.linspace(0.2, 1.0, 401)
    for df, color, name in ((unknown, UNKNOWN, f"never-heard ships ({len(unknown)})"),
                            (known, KNOWN, f"known ships ({len(known):,})")):
        a.plot(x, 100 * (df.conf.to_numpy()[:, None] < x).mean(0), color=color, lw=3.5, label=name)
    thr = res.threshold_ship.mean()
    a.axvline(thr, color=INK, ls="--", lw=2)
    a.text(thr + 0.012, 92, f"threshold ≈ {thr:.2f}", color=INK, fontsize=13)
    at_u, at_k = 100 * unknown.caught.mean(), 100 * known.caught.mean()
    a.annotate(f"{at_u:.0f} % of never-heard ships", (thr, at_u), (0.21, at_u + 12), color=UNKNOWN, zorder=4,
               bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1),
               fontsize=13, arrowprops=dict(arrowstyle="-", color=UNKNOWN))
    a.annotate(f"{at_k:.0f} % of known ships", (thr, at_k), (0.21, at_k + 8), color=KNOWN, fontsize=13, zorder=4,
               bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1),
               arrowprops=dict(arrowstyle="-", color=KNOWN))
    a.set_xlim(0.2, 1.0); a.set_ylim(0, 100)
    a.set_xlabel("ship confidence (highest class probability, clips averaged)")
    a.set_ylabel("ships below this confidence (%)")
    a.legend(loc="lower right", frameon=False, fontsize=13)
    a.set_title("Below the threshold = called \"unknown\"", loc="left", color=INK, fontsize=17)

    t = unknown.groupby("label").agg(ships=("caught", "size"), caught=("caught", "sum"))
    t["share"] = 100 * t.caught / t.ships
    t = t.sort_values("share")
    y = np.arange(len(t))
    b.barh(y, t.share, 0.6, color=UNKNOWN, edgecolor=SURFACE)
    for yy, row in zip(y, t.itertuples()):
        b.text(row.share + 1.5, yy, f"{row.share:.0f} %  ({row.caught} of {row.ships})", va="center", color=INK,
               fontsize=13, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1), zorder=3)
    b.set_yticks(y, [TYPES[i] for i in t.index], color=INK)
    b.set_xlim(0, 100)
    b.set_xlabel("never-heard ships called \"unknown\" (%)")
    total = unknown.caught.mean() * 100
    b.axvline(total, color=MUTED, ls=":", lw=2, zorder=0)
    b.text(total + 1, -0.75, f"all types: {total:.0f} %", color=MUTED, fontsize=13)
    b.set_ylim(-0.95, len(t) - 0.5)
    b.set_title("Caught, by type (5 folds pooled)", loc="left", color=INK, fontsize=17)
    fig.tight_layout()
    path = OUT / "26_unknown_cnn_per_ship.png"
    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", path)


if __name__ == "__main__":
    main()
