"""Slide 9 (walk) chart: validation accuracy after each step of the stricter walk, with each step's gain.

  python src/feature_walk2.py --judge lr               # -> runs/feature_walk2/{steps.jsonl, result_lr.json}
  python docs/review3-figures/make_walk2_path_figure.py # -> docs/review3-figures/readable/09_walk_path.png

One row per step of the walk (first pass, the one re-check that dropped a set, and the second pass where something
changed; the second-pass retries of mel mean + spread and loudness and the three pairs, all left out, are only in the
notes). Filled dots: the model after the step (the line joins them). Hollow dots: a set tried and left out.
Right column: the step's gain, mean over the 5 validation folds +/- 2 standard errors.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "readable"
INK, MUTED, GRID, SURFACE = "#10283d", "#5b7083", "#d9e1e8", "#ffffff"
BLUE, GREY, ORANGE = "#0f6fff", "#b8c4ce", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 18, "ytick.labelsize": 19,
                     "axes.labelsize": 19, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
NAMES = ["mel mean + spread", "mel percentiles", "MFCC", "1/3-octave levels", "level changes over time",
         "spectral shape", "loudness and impulsiveness", "tonal lines", "propeller modulation (DEMON)"]
SHORT = {"mel percentiles": "mel percentiles", "mel mean + spread": "mel mean + spread",
         "1/3-octave levels": "1/3-octave levels", "spectral shape": "spectral shape",
         "loudness and impulsiveness": "loudness & impulsiveness", "MFCC": "MFCC",
         "propeller modulation (DEMON)": "propeller rate (DEMON)", "tonal lines": "tonal lines",
         "level changes over time": "level changes"}


def canon(sets):
    return " | ".join(sorted(set(sets), key=NAMES.index))


def parse(s):
    out = []
    while s:
        n = next(n for n in sorted(NAMES, key=len, reverse=True) if s.startswith(n))
        out.append(n)
        s = s[len(n):].removeprefix(" + ")
    return out


def steps():
    walk = ROOT / "runs" / "feature_walk2"
    rows = {}
    for line in (walk / "steps.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["judge"] == "lr":
            rows[r["groups"]] = 100 * np.array(r["val_by_fold"])
    res = json.loads((walk / "result_lr.json").read_text())
    kept = [res["order"][0]]
    out = [dict(label=f"start: {SHORT[kept[0]]}", acc=rows[canon(kept)].mean(), model=True, gain=None)]
    for h in res["history"]:
        sets = parse(h["sets"])
        if h["stage"] == "pairs":
            continue
        if h["action"] == "add":
            trial = kept + sets
            d = rows[canon(trial)] - rows[canon(kept)]
            ok = h["decision"] == "kept"
            if h["stage"] == "second pass" and not ok and sets[0] in ("mel mean + spread", "loudness and impulsiveness"):
                kept_acc = None   # left out again; shown in the notes only
            else:
                prefix = "2nd pass: + " if h["stage"] == "second pass" else "+ "
                out.append(dict(label=prefix + SHORT[sets[0]], acc=rows[canon(trial)].mean(), model=ok,
                                gain=(d.mean(), 2 * d.std(ddof=1) / np.sqrt(5))))
            if ok:
                kept = trial
        else:
            trial = [x for x in kept if x not in sets]
            d = rows[canon(kept)] - rows[canon(trial)]
            out.append(dict(label=f"re-check: − {SHORT[sets[0]]}", acc=rows[canon(trial)].mean(), model=True,
                            gain=(d.mean(), 2 * d.std(ddof=1) / np.sqrt(5)), drop=True))
            kept = trial
    out[[i for i, s in enumerate(out) if s["model"]][-1]]["final"] = True
    return out


def main():
    data = steps()
    fig, ax = plt.subplots(figsize=(16, 6.3))
    y = np.arange(len(data))[::-1]
    path = [(s["acc"], yy) for s, yy in zip(data, y) if s["model"]]
    ax.plot([p[0] for p in path], [p[1] for p in path], color=BLUE, lw=3, zorder=1)
    for s, yy in zip(data, y):
        if s["model"]:
            ax.scatter(s["acc"], yy, s=200, color=BLUE, zorder=3, edgecolor=SURFACE, lw=2)
            txt = f"{s['acc']:.1f} %"
            if s.get("drop"):                       # the line comes back from the right: label on the left
                ax.text(s["acc"] - 0.15, yy, txt, fontsize=18, color=INK, va="center", ha="right")
            else:
                ax.text(s["acc"] + 0.12, yy + 0.08, txt, fontsize=18, color=INK, va="bottom", ha="left",
                        weight="bold" if s.get("final") else "normal")
        else:
            ax.scatter(s["acc"], yy, s=170, facecolor=SURFACE, edgecolor=GREY, lw=2.5, zorder=3)
            ax.text(s["acc"] - 0.12, yy, f"left out {s['acc']:.1f} %", fontsize=16, color=MUTED, va="center", ha="right")
        if s["gain"] is not None:
            m, e = s["gain"]
            if s.get("drop"):
                word, col = "adds only", ORANGE
            else:
                word, col = ("kept", INK) if s["model"] else ("left out", MUTED)
            ax.text(1.005, yy, f"{m:+.2f} ± {e:.2f}", transform=ax.get_yaxis_transform(), fontsize=17,
                    color=col if s.get("drop") else (INK if s["model"] else MUTED), va="center", ha="left",
                    family="DejaVu Sans Mono")
    ax.text(1.005, len(data) - 0.35, "gain ± 2 SE", transform=ax.get_yaxis_transform(), fontsize=16, color=MUTED,
            va="bottom", ha="left")
    ax.set_yticks(y, [s["label"] for s in data])
    for lab, s in zip(ax.get_yticklabels(), data):
        lab.set_color(INK if s["model"] else MUTED)
        if s.get("final"):
            lab.set_fontweight("bold")
    ax.set_xlim(44.9, 53.2)
    ax.set_ylim(-0.6, len(data) - 0.3)
    ax.set_xlabel("validation balanced accuracy (%), 6 classes + other · keep a set only if it adds ≥ 0.5 points")
    ax.grid(axis="x", color=GRID, lw=1, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "09_walk_path.png", dpi=100, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", OUT / "09_walk_path.png")


if __name__ == "__main__":
    main()
