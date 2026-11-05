"""Readable versions of the review-3 slide charts (slides 9 and 18-22), from the saved results.

  python docs/review3-figures/make_readable_figures.py     # -> docs/review3-figures/readable/*.png

The user found the late charts hard to grasp. Rules here: one message per chart, big type (22-26 pt at
1600 px wide, so about 28 px on the slide), values written on the marks, no chart titles (the slide has one),
confusion matrices replaced by "share right per bucket" bars. Light background, deck blue as the one accent.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
OUT = Path(__file__).resolve().parent / "readable"
INK, MUTED, GRID, SURFACE = "#10283d", "#5b7083", "#d9e1e8", "#ffffff"
BLUE, LIGHT, ORANGE, RED, GREY = "#0f6fff", "#9cc2ff", "#eb6834", "#b03a2e", "#b8c4ce"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 22, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 21, "ytick.labelsize": 21,
                     "axes.labelsize": 22, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "legend.fontsize": 20,
                     "legend.frameon": False})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=100, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / name)


def jsonl(path):
    return pd.DataFrame([json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()])


# ---------------------------------------------------------------- slide 9: feature sets added one at a time
SHORT = {"mel percentiles": "mel percentiles", "mel mean + spread": "mel mean + spread",
         "1/3-octave levels": "1/3-octave levels", "spectral shape": "spectral shape",
         "loudness and impulsiveness": "loudness & impulsiveness", "MFCC": "MFCC",
         "propeller modulation (DEMON)": "propeller rate (DEMON)", "tonal lines": "tonal lines",
         "level changes over time": "level changes"}


def fig_walk():
    w = jsonl(RUNS / "with_other" / "walk.jsonl")
    res = json.loads((RUNS / "with_other" / "walk_result.json").read_text())
    first = res["order"][0]
    steps = [(f"start: {SHORT[first]}", 100 * res["alone"][first], True)]
    for r in w[w.stage == "walk"].sort_values("step").itertuples():
        steps.append((f"+ {SHORT[r.added]}", 100 * r.val, r.val > r.before))
    fig, ax = plt.subplots(figsize=(16, 8))
    y = np.arange(len(steps))[::-1]
    kept_vals = [v for _, v, k in steps if k]
    ax.plot(kept_vals, [yy for yy, (_, _, k) in zip(y, steps) if k], color=BLUE, lw=3, zorder=1)
    for yy, (name, v, kept) in zip(y, steps):
        ax.scatter(v, yy, s=260, zorder=3, color=BLUE if kept else SURFACE, edgecolor=BLUE if kept else GREY, lw=3)
        if kept:
            ax.text(v + 0.25, yy, f"{v:.1f} %", va="center", fontsize=21, color=INK)
        else:
            ax.text(v - 0.25, yy, f"dropped  {v:.1f} %", va="center", ha="right", fontsize=21, color=MUTED)
    ax.set_yticks(y, [n for n, _, _ in steps])
    for lab, (_, _, kept) in zip(ax.get_yticklabels(), steps):
        lab.set_color(INK if kept else MUTED)
    ax.set_xlim(43.5, 54)
    ax.set_xlabel("validation balanced accuracy (%), 6 classes + other")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID)
    save(fig, "09_walk.png")


# ---------------------------------------------------------------- slide 18: unknown from low confidence (GB, lesson 21)
def fig_left_out():
    r = pd.DataFrame([x for x in json.loads((RUNS / "open_set" / "results.json").read_text())["results"]
                      if x["experiment"] == "B"])
    fig, ax = plt.subplots(figsize=(10, 6))
    vals = [("per clip", 100 * r.auroc_clip), ("per ship", 100 * r.auroc_ship)]
    for i, (name, v) in enumerate(vals):
        ax.barh(i, v.mean(), 0.55, color=LIGHT if i == 0 else BLUE)
        ax.scatter(v, np.full(len(v), i), color=INK, s=40, zorder=3)
        ax.text(v.max() + 1.5, i, f"{v.mean():.0f} %", va="center", fontsize=26, weight="bold", color=INK)
    ax.axvline(50, color=MUTED, ls="--", lw=2)
    ax.text(50.5, 1.45, "50 % = no help", color=MUTED, fontsize=19)
    ax.set_yticks([0, 1], [n for n, _ in vals]); ax.set_xlim(40, 100); ax.set_ylim(-0.5, 1.7)
    ax.set_xlabel("AUROC: known vs never-heard (%)\ndots = the 5 test folds")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    save(fig, "18_unknown_gb.png")


# ---------------------------------------------------------------- slide 19: unknown ships with the CNN, per ship
TYPES = {"sailing": "sailing", "passengership": "passenger ship", "other": "other (dataset)", "rescue": "rescue",
         "dredger": "dredger", "pilotvessel": "pilot vessel"}


def fig_unknown_cnn():
    res = pd.DataFrame(json.loads((RUNS / "unknown_cnn" / "results.json").read_text())["results"]).set_index("fold")
    p = pd.read_parquet(RUNS / "unknown_cnn" / "predictions.parquet")
    cols = [c for c in p.columns if c.startswith("p_")]
    s = (p[p.split.isin(["test", "unknown"])].groupby(["fold", "split", "ship"])
         .agg(**{c: (c, "mean") for c in cols}, label=("label", "first")).reset_index())
    s["conf"] = s[cols].max(axis=1)
    s["caught"] = s.conf < s.fold.map(res.threshold_ship)
    known, unknown = s[s.split == "test"], s[s.split == "unknown"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(18, 7.2), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.45})
    x = np.linspace(0.2, 1.0, 401)
    for df, color, name in ((unknown, RED, "never-heard ships"), (known, BLUE, "known ships")):
        a.plot(x, 100 * (df.conf.to_numpy()[:, None] < x).mean(0), color=color, lw=4.5, label=name)
    thr = res.threshold_ship.mean()
    a.axvline(thr, color=INK, ls="--", lw=2.5)
    at_u, at_k = 100 * unknown.caught.mean(), 100 * known.caught.mean()
    a.scatter([thr], [at_u], s=110, color=RED, zorder=4)
    a.text(thr - 0.015, at_u + 3, f"{at_u:.0f} % of\nnever-heard", ha="right", va="bottom", color=RED,
           fontsize=22, weight="bold")
    a.annotate(f"{at_k:.0f} % of known ships", (thr, at_k), (0.6, 6), color=BLUE, fontsize=22, weight="bold",
               arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.5))
    a.text(thr + 0.012, 62, "threshold", color=INK, fontsize=20)
    a.set_xlim(0.2, 1.0); a.set_ylim(0, 100)
    a.set_xlabel("ship confidence")
    a.set_ylabel("ships called \"unknown\" (%)")
    a.legend(loc="upper left")
    t = unknown.groupby("label").agg(ships=("caught", "size"), caught=("caught", "sum"))
    t["share"] = 100 * t.caught / t.ships
    t = t.sort_values("share")
    y = np.arange(len(t))
    b.barh(y, t.share, 0.62, color=RED)
    for yy, row in zip(y, t.itertuples()):
        b.text(row.share + 2, yy, f"{row.share:.0f} %  ({row.caught} of {row.ships})", va="center", fontsize=21,
               color=INK)
    b.set_yticks(y, [TYPES[i] for i in t.index]); b.set_xlim(0, 100)
    b.set_xlabel("never-heard ships caught (%)")
    b.spines["left"].set_visible(False); b.tick_params(axis="y", length=0)
    save(fig, "19_unknown_cnn.png")


# ---------------------------------------------------------------- slide 20: "other" always a class, five feature sets
def fig_other_sets():
    m = jsonl(RUNS / "with_other" / "main.jsonl")
    g = (100 * m.groupby(["features", "model"]).balanced_accuracy.mean()).unstack("model")
    names = {"mel mean + spread (257)": "mel mean + spread", "mel mean + spread + percentiles (641)":
             "mel + percentiles", "engineered (144)": "engineered (144)",
             "engineered + mel mean + spread (401)": "engineered + mel (401)",
             "engineered + mel mean + spread + percentiles (785)": "engineered + mel + percentiles"}
    g = g.loc[[k for k in names if k in g.index]].sort_values("gradient boosting")
    fig, ax = plt.subplots(figsize=(16, 7.5))
    y = np.arange(len(g))
    for j, (model, color, lab) in enumerate((("gradient boosting", BLUE, "gradient boosting"),
                                             ("logistic regression", LIGHT, "logistic regression"))):
        off = 0.2 if j == 0 else -0.2
        ax.barh(y + off, g[model], 0.38, color=color, label=lab)
        for yy, v in zip(y, g[model]):
            ax.text(v + 0.4, yy + off, f"{v:.1f}", va="center", fontsize=20, color=INK)
    ax.axvline(100 / 7, color=MUTED, ls="--", lw=2)
    ax.text(100 / 7 + 0.5, len(g) - 0.45, "chance 14.3 %", color=MUTED, fontsize=19)
    ax.set_yticks(y, [names[i] for i in g.index]); ax.set_xlim(0, 64)
    ax.set_xlabel("balanced accuracy per clip (%), 6 classes + other, 5 test folds")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.13))
    save(fig, "20_other_sets.png")


# ---------------------------------------------------------------- slides 21-22: share right per bucket
def bucket_bars(ax, pred, buckets, xlabel):
    true, clf = pred.true_bucket.to_numpy(), pred.clf_bucket.to_numpy()
    exact = np.array([(clf[true == b] == b).mean() for b in range(len(buckets))]) * 100
    near = np.array([(np.abs(clf[true == b] - b) == 1).mean() for b in range(len(buckets))]) * 100
    x = np.arange(len(buckets))
    ax.bar(x, exact, 0.7, color=BLUE, label="exact bucket")
    ax.bar(x, near, 0.7, bottom=exact, color=LIGHT, label="next bucket")
    for xx, e in zip(x, exact):
        if e >= 15:
            ax.text(xx, e / 2, f"{e:.0f} %", ha="center", va="center", color=SURFACE, fontsize=22, weight="bold")
        else:
            ax.text(xx, e + 2, f"{e:.0f} %", ha="center", va="bottom", color=INK, fontsize=22, weight="bold")
    ax.axhline(100 / len(buckets), color=MUTED, ls="--", lw=2, label=f"chance ({100 / len(buckets):.0f} %)")
    ax.set_xticks(x, buckets); ax.set_ylim(0, 100)
    ax.set_ylabel("clips (%)"); ax.set_xlabel(xlabel)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.13), fontsize=19)
    return exact


def fig_distance():
    d = pd.read_parquet(RUNS / "distance_classical_400m" / "predictions.parquet")
    buckets = json.loads((RUNS / "distance_classical_400m" / "results.json").read_text())["buckets"]
    fig, ax = plt.subplots(figsize=(11, 7.5))
    exact = bucket_bars(ax, d, [b.replace(" m", "") for b in buckets], "true distance (m)")
    print("distance exact per bucket:", exact.round(1), "overall", round(100 * (d.clf_bucket == d.true_bucket).mean(), 1))
    save(fig, "21_distance.png")


def fig_distance_matrix():
    """Slide 21 (user's choice): the confusion matrix, made readable: big numbers, rows add up to 100 %,
    the diagonal (exact bucket) outlined, light cells in light grey text."""
    d = pd.read_parquet(RUNS / "distance_classical_400m" / "predictions.parquet")
    buckets = [b.replace(" m", "") for b in json.loads(
        (RUNS / "distance_classical_400m" / "results.json").read_text())["buckets"]]
    n = len(buckets)
    cm = np.zeros((n, n))
    for t, p in zip(d.true_bucket, d.clf_bucket):
        cm[int(t), int(p)] += 1
    cm = 100 * cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=80)
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            ax.text(j, i, f"{v:.0f} %", ha="center", va="center", fontsize=24,
                    weight="bold" if i == j else "normal",
                    color=SURFACE if v >= 45 else (INK if v >= 8 else MUTED))
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=3))
    ax.set_xticks(range(n), [b.replace("-", "-\n") for b in buckets])
    ax.set_yticks(range(n), buckets)
    ax.set_xlabel("predicted distance (m)")
    ax.set_ylabel("true distance (m)")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    print("distance diagonal:", np.diag(cm).round(1))
    save(fig, "21_distance_matrix.png")


def matrix_panel(ax, pred, labels, axis_name):
    n = len(labels)
    cm = np.zeros((n, n))
    for t, p in zip(pred.true_bucket, pred.clf_bucket):
        cm[int(t), int(p)] += 1
    cm = 100 * cm / cm.sum(1, keepdims=True)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=80)
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            ax.text(j, i, f"{v:.0f} %", ha="center", va="center", fontsize=22,
                    weight="bold" if i == j else "normal",
                    color=SURFACE if v >= 45 else (INK if v >= 8 else MUTED))
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=3))
    ax.set_xticks(range(n), [b.replace("-", "-\n") if len(b) > 5 else b for b in labels])
    ax.set_yticks(range(n), labels)
    ax.set_xlabel(f"predicted {axis_name}")
    ax.set_ylabel(f"true {axis_name}")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return np.diag(cm)


def scatter_panel(ax, true, pred, top, unit, ticks):
    rng = np.random.default_rng(0)
    keep = rng.permutation(len(true))[:6000]                 # 6,000 random clips: readable, same picture
    ax.scatter(true[keep], pred[keep], s=10, alpha=0.25, color=BLUE, edgecolor="none", rasterized=True,
               label="one test clip")
    ax.plot([0, top], [0, top], color=INK, lw=3, label="perfect prediction")
    ax.set_xlim(0, top); ax.set_ylim(0, top)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xlabel(f"true {unit}"); ax.set_ylabel(f"predicted {unit}")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", markerscale=3)
    err = np.abs(pred - true)
    return np.median(err), err.mean()


def fig_distance_both():
    d = pd.read_parquet(RUNS / "distance_classical_400m" / "predictions.parquet")
    buckets = [b.replace(" m", "") for b in json.loads(
        (RUNS / "distance_classical_400m" / "results.json").read_text())["buckets"]]
    fig, (a, b) = plt.subplots(1, 2, figsize=(19, 8.2), gridspec_kw={"wspace": 0.35})
    diag = matrix_panel(a, d, buckets, "distance (m)")
    med, mean = scatter_panel(b, d.true_m.to_numpy(), d.reg_m.to_numpy(), 2000, "distance (m)",
                              [0, 500, 1000, 1500, 2000])
    print("distance diagonal", diag.round(1), "regression median / mean error", round(med), round(mean))
    save(fig, "21_distance_matrix_regression.png")


def fig_speed_both():
    s = pd.read_parquet(RUNS / "speed_classical_4kn" / "predictions.parquet")
    buckets = [x.replace(" kn", "") for x in json.loads(
        (RUNS / "speed_classical_4kn" / "results.json").read_text())["buckets"]]
    fig, (a, b) = plt.subplots(1, 2, figsize=(19, 8.2), gridspec_kw={"wspace": 0.35})
    diag = matrix_panel(a, s, buckets, "speed (knots)")
    med, mean = scatter_panel(b, s.true_m.to_numpy(), s.reg_m.to_numpy(), 25, "speed (knots)",
                              [0, 5, 10, 15, 20, 25])
    print("speed diagonal", diag.round(1), "regression median / mean error", round(med, 2), round(mean, 2),
          "within 4 kn", round(100 * (np.abs(s.reg_m - s.true_m) <= 4).mean(), 1))
    save(fig, "22_speed_matrix_regression.png")


def fig_open_set_matrix():
    """Slide after 19: per ship, 5 known classes + never-heard ships (rows) against the 5 classes + "unknown"
    (columns). Each fold's own threshold (95 % of known validation ships kept); five folds pooled."""
    res = pd.DataFrame(json.loads((RUNS / "unknown_cnn" / "results.json").read_text())["results"]).set_index("fold")
    p = pd.read_parquet(RUNS / "unknown_cnn" / "predictions.parquet")
    cols = [c for c in p.columns if c.startswith("p_")]
    names = [c[2:] for c in cols]
    s = (p[p.split.isin(["test", "unknown"])].groupby(["fold", "split", "ship"])
         .agg(**{c: (c, "mean") for c in cols}, label=("label", "first")).reset_index())
    conf, arg = s[cols].max(axis=1).to_numpy(), s[cols].to_numpy().argmax(1)
    pred = np.where(conf < s.fold.map(res.threshold_ship).to_numpy(), len(names), arg)
    merged = {"cargo": "cargo + tanker", "tanker": "cargo + tanker"}
    true_name = np.where(s.split == "unknown", "never heard",
                         s.label.map(lambda l: merged.get(l, l)).to_numpy())
    rows = names + ["never heard"]
    true = np.array([rows.index(t) for t in true_name])
    k = len(rows)
    counts = np.zeros((k, k))
    for t, q in zip(true, pred):
        counts[t, q] += 1
    pct = 100 * counts / counts.sum(1, keepdims=True)
    short = {"background": "background", "cargo + tanker": "cargo +\ntanker", "fishing": "fishing",
             "pleasurecraft": "pleasure\ncraft", "tug": "tug", "never heard": "never\nheard", "unknown": "\"unknown\""}
    fig, ax = plt.subplots(figsize=(13, 10.5))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=90)
    for i in range(k):
        for j in range(k):
            v = pct[i, j]
            ax.text(j, i, f"{v:.0f} %", ha="center", va="center", fontsize=23,
                    weight="bold" if i == j else "normal",
                    color=SURFACE if v >= 50 else (INK if v >= 8 else MUTED))
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=3))
    ax.axhline(k - 1.5, color=RED, lw=3); ax.axvline(k - 1.5, color=RED, lw=3)
    xshort = {**short, "background": "back-\nground"}
    ax.set_xticks(range(k), [xshort[n] for n in names] + [xshort["unknown"]])
    ax.set_yticks(range(k), [f"{short[r]} ({int(counts[i].sum())})" for i, r in enumerate(rows)])
    ax.set_xlabel("answer, per ship")
    ax.set_ylabel("true type (ships)")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    print("rows (ships):", dict(zip(rows, counts.sum(1).astype(int))))
    print("diagonal %:", dict(zip(rows, np.diag(pct).round(1))))
    print("known ships called unknown %:", round(100 * counts[:-1, -1].sum() / counts[:-1].sum(), 1),
          "| never-heard called known: ", dict(zip(names, pct[-1, :-1].round(1))))
    save(fig, "19b_open_set_matrix.png")


def fig_speed():
    s = pd.read_parquet(RUNS / "speed_classical_4kn" / "predictions.parquet")
    buckets = json.loads((RUNS / "speed_classical_4kn" / "results.json").read_text())["buckets"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(19, 7.2), gridspec_kw={"wspace": 0.3})
    exact = bucket_bars(a, s, [x.replace(" kn", "") for x in buckets], "true speed (knots)")
    print("speed exact per bucket:", exact.round(1), "overall", round(100 * (s.clf_bucket == s.true_bucket).mean(), 1))
    bins = np.arange(0, 21, 1)
    mid = (bins[:-1] + bins[1:]) / 2
    grp = s.groupby(pd.cut(s.true_m, bins), observed=False).reg_m
    q = grp.quantile([0.25, 0.5, 0.75]).unstack()
    n = grp.size().to_numpy()
    ok = n >= 30
    b.fill_between(mid[ok], q[0.25].to_numpy()[ok], q[0.75].to_numpy()[ok], color=LIGHT, alpha=0.7,
                   label="middle half of predictions")
    b.plot(mid[ok], q[0.5].to_numpy()[ok], color=BLUE, lw=4, label="median prediction")
    b.plot([0, 20], [0, 20], color=MUTED, ls="--", lw=2, label="perfect")
    b.set_xlim(0, 20); b.set_ylim(0, 20); b.set_yticks([0, 5, 10, 15, 20])
    b.set_xlabel("true speed (knots)"); b.set_ylabel("predicted speed (knots)")
    b.legend(loc="upper left")
    save(fig, "22_speed.png")


if __name__ == "__main__":
    fig_walk()
    fig_left_out()
    fig_unknown_cnn()
    fig_other_sets()
    fig_distance()
    fig_speed()
