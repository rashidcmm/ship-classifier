"""How accuracy grows with listening time: 1, 2, 3, ... clips of the same passage averaged (no training).

"Per ship" (lesson 13: 70.5 %) averages ALL clips of a ship, but that is 1-2 clips (5-10 s) for a cargo ship and
15-16 (75-80 s) for a tanker. For a system in the water the fair question is: after listening to one contact for
t seconds, how often is the answer right? Here every test passage (one ship on one day; a background recording day)
is cut into consecutive blocks of n clips in time order, the class probabilities of a block are averaged, and the
block is scored. The clips are 5 s each but not back to back (the dataset samples a passage, often one clip every
15 s), so both the audio heard (n x 5 s) and the time from first to last clip are reported.

Only passages with at least n clips count at n, so the classes drop out as n grows (cargo after 2 clips). The main
curve therefore uses the four classes that have long passages (fishing, pleasure craft, tanker, tug), up to 9 clips.

  python src/listening_time.py        # -> runs/listening_time/{blocks.csv, summary.csv}
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_DIR, RUNS_DIR

OUT_DIR = RUNS_DIR / "listening_time"
N_CLIPS = (1, 2, 3, 4, 6, 9, 12)
LONG = ("fishing", "pleasurecraft", "tanker", "tug")      # passages of 9+ clips
MAX_GAP_S = 600                                           # a gap of more than 10 min starts a new passage
SOURCES = {
    "6 classes (lesson 13)": dict(path=RUNS_DIR / "six_classes" / "predictions.parquet",
                                  where={"features": "engineered", "model": "gradient boosting"}),
    "6 classes + other (lesson 22)": dict(path=RUNS_DIR / "with_other" / "main_predictions.parquet",
                                          where={"grouping": "6 classes + other",
                                                 "features": "engineered + mel mean + spread (401)",
                                                 "model": "gradient boosting"}),
}


def names_of(pred: pd.DataFrame) -> list[str]:
    return [c[2:] for c in pred.columns if c.startswith("p_")]


def clip_table(data_dir=DATA_DIR) -> pd.DataFrame:
    t = pd.read_parquet(Path(data_dir) / "clips.parquet")
    start = pd.to_datetime(t.path.str.extract(r"(\d{8}T\d{6})")[0], format="%Y%m%dT%H%M%S")
    ship = t.MMSI.map(lambda m: f"mmsi{m:.0f}").where(t.label != "background", "day" + t.date.astype(str))
    out = pd.DataFrame({"label": t.label, "ship": ship, "start": start}).sort_values(["ship", "start"])
    new = (out.ship != out.ship.shift()) | (out.start.diff().dt.total_seconds() > MAX_GAP_S)
    out["passage"] = new.cumsum()
    return out


def blocks(pred: pd.DataFrame, clips: pd.DataFrame, n: int) -> pd.DataFrame:
    """One row per block of n consecutive clips of a passage: true class, right or wrong, seconds first to last."""
    pcols = [c for c in pred.columns if c.startswith("p_")]
    names = [c[2:] for c in pcols]
    d = pred[["row", *pcols]].merge(clips, left_on="row", right_index=True).sort_values(["passage", "start"])
    d["pos"] = d.groupby("passage").cumcount()
    size = d.groupby("passage").row.transform("size")
    d = d[d.pos < size - size % n]                       # drop the remainder that doesn't fill a block
    d["block"] = d.passage.astype(str) + "_" + (d.pos // n).astype(str)
    g = d.groupby("block")
    b = g[pcols].mean()
    b["label"] = g.label.first()
    b["pred"] = np.array(names)[b[pcols].to_numpy().argmax(1)]
    b["right"] = b.pred == b.label
    b["span_s"] = (g.start.max() - g.start.min()).dt.total_seconds() + 5
    return b[["label", "pred", "right", "span_s"]].assign(n=n)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clips_all = clip_table()
    all_blocks, rows = [], []
    for source, spec in SOURCES.items():
        pred = pd.read_parquet(spec["path"])
        for k, v in spec["where"].items():
            pred = pred[pred[k] == v]
        names = names_of(pred)
        if "other" in names:            # the left-out types (sailing, passenger ship, ...) are one class "other"
            clips = clips_all.assign(label=clips_all.label.where(clips_all.label.isin(names), "other"))
        else:
            clips = clips_all
        for n in N_CLIPS:
            b = blocks(pred, clips, n).assign(source=source)
            all_blocks.append(b)
            recall = b.groupby("label").right.mean()
            long = b[b.label.isin(LONG)]
            have_all = long.label.nunique() == len(LONG)
            rows.append({"source": source, "n_clips": n, "audio_s": 5 * n,
                         "balanced_all": float(recall.mean()) if len(recall) == len(names_of(pred)) else np.nan,
                         "median_span_s": float(long.span_s.median()) if len(long) else np.nan,
                         "balanced_long4": float(long.groupby("label").right.mean().mean()) if have_all else np.nan,
                         "blocks_long4": len(long),
                         **{f"recall_{c}": float(recall.get(c, np.nan)) for c in sorted(b.label.unique())},
                         **{f"blocks_{c}": int((b.label == c).sum()) for c in sorted(b.label.unique())}})
        # one answer per whole passage (one encounter) and per ship (all its passages, as lesson 13's "per ship")
        pcols, names = [f"p_{c}" for c in names_of(pred)], np.array(names_of(pred))
        d = pred[["row", *pcols]].merge(clips, left_on="row", right_index=True)
        for unit in ("passage", "ship"):
            g = d.groupby(unit)
            right = pd.Series(names[g[pcols].mean().to_numpy().argmax(1)] == g.label.first().to_numpy(),
                              index=g.label.first().to_numpy())
            rec = right.groupby(level=0).mean()
            rows.append({"source": source, "n_clips": f"whole {unit}", "balanced_all": float(rec.mean()),
                         "balanced_long4": float(rec[list(LONG)].mean()),
                         **{f"recall_{c}": float(v) for c, v in rec.items()},
                         **{f"blocks_{c}": int(v) for c, v in right.groupby(level=0).size().items()}})
    pd.concat(all_blocks).to_csv(OUT_DIR / "blocks.csv")
    s = pd.DataFrame(rows)
    s.to_csv(OUT_DIR / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    for source, g in s.groupby("source", sort=False):
        print(f"\n{source}")
        show = g.set_index("n_clips")
        cols = ["audio_s", "median_span_s", "balanced_all", "balanced_long4", "blocks_long4"] + \
               [c for c in show.columns if c.startswith("recall_")]
        t = show[cols].T
        pct = t.index.str.startswith(("recall", "balanced"))
        t.loc[pct] = t.loc[pct] * 100
        print(t.round(1).to_string())


if __name__ == "__main__":
    main()
