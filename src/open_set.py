"""Ship types we left out, and PANN embeddings (lesson 21).

Since lesson 13 six of the dataset's twelve classes were left out: sailing, other, passenger ship, rescue, dredger,
pilot vessel (about 520 ships). A real system will hear them too. Three experiments, same folds for the six known
classes as lessons 13-20 (the left-out ships get their own five folds, grouped by ship, so they too are tested on
ships never heard):

  A. an "other" class: all left-out types as one more class (5 classes + other, 4 classes + other);
  B. "unknown": train on the five classes only and answer "unknown" when the model is not sure; the threshold keeps
     95 % of the known validation clips, and the left-out test ships show how many unknown ships it catches;
  C. PANN embeddings (CNN14 pretrained on AudioSet, 2,048 numbers per clip, reduced to 256 by PCA fitted on the
     training + validation clips): alone, and next to the 144 engineered features, on the five classes.

Every merged class gets the same total weight; inside it each original class the same; inside that each ship.

  python src/open_set.py      # -> runs/open_set/ (CPU, about 30 minutes; needs src/panns_features.py first)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import merged_classes as M
import panns_features as P
import six_classes as S
from keepawake import keep_awake
from paths import DATA_DIR, RUNS_DIR

OUT_DIR = RUNS_DIR / "open_set"
LEFT_OUT = ["sailing", "other", "passengership", "rescue", "dredger", "pilotvessel"]
FIVE = {"background": ["background"], "cargo + tanker": ["cargo", "tanker"], "fishing": ["fishing"],
        "pleasurecraft": ["pleasurecraft"], "tug": ["tug"]}
FOUR = {"background": ["background"], "cargo + tanker": ["cargo", "tanker"], "fishing + tug": ["fishing", "tug"],
        "pleasurecraft": ["pleasurecraft"]}
GROUPINGS_A = {"5 classes + other": {**FIVE, "other": LEFT_OUT}, "4 classes + other": {**FOUR, "other": LEFT_OUT}}
KEEP_KNOWN = 0.95                     # B: threshold keeps this share of the known validation clips
N_PCA = 256


def all_folds(table, seed=0) -> np.ndarray:
    """Lesson 13's folds for the six classes, unchanged; the left-out classes get their own five folds by ship."""
    folds = S.make_folds(table, seed=seed)
    keep = np.flatnonzero(table.label.isin(LEFT_OUT).to_numpy())
    sub = table.iloc[keep]
    splitter = StratifiedGroupKFold(S.N_FOLDS, shuffle=True, random_state=seed)
    for k, (_, test) in enumerate(splitter.split(sub, sub.label, S.ship_key(sub)), start=1):
        folds[keep[test]] = k
    return folds


def group_weights(labels, ships, grouping) -> np.ndarray:
    """Every group the same total, every original class the same inside it, every ship the same inside that."""
    w = S.ship_weights(labels, ships)
    size = {l: len(m) for m in grouping.values() for l in m}
    w = w / np.array([size[l] for l in labels])
    return w / w.mean()


class GroupFold:
    """Rows, labels, ships and weights of one fold for any grouping of the twelve labels."""

    def __init__(self, table, folds, k, grouping):
        labels = table.label.to_numpy()
        inside = np.isin(labels, [l for m in grouping.values() for l in m])
        self.k, self.rows = k, S.split_rows(np.where(inside, folds, 0), k)
        self.rows["train+validation"] = np.sort(np.concatenate([self.rows["train"], self.rows["validation"]]))
        index = {l: i for i, m in enumerate(grouping.values()) for l in m}
        ships = S.ship_key(table).to_numpy()
        self.label = {s: labels[r] for s, r in self.rows.items()}
        self.y = {s: np.array([index[l] for l in self.label[s]]) for s in self.rows}
        self.ships = {s: ships[r] for s, r in self.rows.items()}
        self.w = {s: group_weights(self.label[s], self.ships[s], grouping) for s in self.rows}

    def X(self, feats, cols, split):
        return feats.loc[self.rows[split], cols].to_numpy()


def load_engineered(data_dir) -> pd.DataFrame:
    _, _, eng = S.load_features(data_dir)
    left = pd.read_parquet(Path(data_dir) / P.LEFTOUT_FILE).set_index("row")
    return pd.concat([eng, left[eng.columns]])


def per_ship_max(probs, ships) -> pd.Series:
    """Confidence per ship: the highest class probability after averaging the ship's clips."""
    return pd.DataFrame(probs).assign(ship=ships).groupby("ship").mean().max(axis=1)


def experiment_a(table, folds, eng, seed, log):
    out, cols = [], list(eng.columns)
    for gname, grouping in GROUPINGS_A.items():
        names = list(grouping)
        for k in range(1, S.N_FOLDS + 1):
            f = GroupFold(table, folds, k, grouping)
            probs, setting, val, _ = S.fit_and_test(f, eng, cols, "gradient boosting", seed)
            out.append({"experiment": "A", "grouping": gname, "fold": k, "setting": setting,
                        "val_balanced_accuracy": val, **M.scores(f.y["test"], probs, f.ships["test"], names)})
            log(f"A {gname}, fold {k}: {out[-1]['balanced_accuracy']:.3f} per clip, "
                f"{out[-1]['per_ship_balanced_accuracy']:.3f} per ship, other {out[-1]['per_class_recall']['other']:.3f}")
    return out


def experiment_b(table, folds, eng, seed, log):
    """Known = the five classes; unknown = left-out test ships of the same fold. Uses the model trained on the
    training folds (its validation clips set the threshold, so they must not be in its training data)."""
    out, cols = [], list(eng.columns)
    labels = table.label.to_numpy()
    ships_all = S.ship_key(table).to_numpy()
    for k in range(1, S.N_FOLDS + 1):
        f = GroupFold(table, folds, k, FIVE)
        model, _, _ = S.fit_gb(f.X(eng, cols, "train"), f.y["train"], f.w["train"], f.X(eng, cols, "validation"),
                               f.y["validation"], f.w["validation"], seed)
        val_conf = model.predict_proba(f.X(eng, cols, "validation")).max(1)
        threshold = float(np.quantile(val_conf, 1 - KEEP_KNOWN))
        unknown_rows = np.flatnonzero(np.isin(labels, LEFT_OUT) & (folds == k))
        p_known = model.predict_proba(f.X(eng, cols, "test"))
        p_unknown = model.predict_proba(eng.loc[unknown_rows, cols].to_numpy())
        c_known, c_unknown = p_known.max(1), p_unknown.max(1)
        s_known, s_unknown = per_ship_max(p_known, f.ships["test"]), per_ship_max(p_unknown, ships_all[unknown_rows])
        caught = pd.Series(c_unknown < threshold).groupby(labels[unknown_rows]).mean()
        out.append({"experiment": "B", "fold": k, "threshold": threshold,
                    "known_accepted": float((c_known >= threshold).mean()),
                    "unknown_rejected": float((c_unknown < threshold).mean()),
                    "auroc_clip": float(roc_auc_score(np.r_[np.zeros(len(c_known)), np.ones(len(c_unknown))],
                                                      -np.r_[c_known, c_unknown])),
                    "auroc_ship": float(roc_auc_score(np.r_[np.zeros(len(s_known)), np.ones(len(s_unknown))],
                                                      -np.r_[s_known, s_unknown])),
                    "rejected_by_type": {t: float(v) for t, v in caught.items()},
                    "known_rejected_by_class": {n: float((c_known[f.y["test"] == i] < threshold).mean())
                                                for i, n in enumerate(FIVE)},
                    "unknown_called_as": {n: float((p_unknown.argmax(1) == i).mean()) for i, n in enumerate(FIVE)}})
        r = out[-1]
        log(f"B fold {k}: known accepted {r['known_accepted']:.3f}, unknown rejected {r['unknown_rejected']:.3f}, "
            f"AUROC clip {r['auroc_clip']:.3f} ship {r['auroc_ship']:.3f}")
    return out


def experiment_c(table, folds, eng, emb, seed, log):
    out = []
    names = list(FIVE)
    for k in range(1, S.N_FOLDS + 1):
        f = GroupFold(table, folds, k, FIVE)
        fit_rows = f.rows["train+validation"]
        pca = PCA(N_PCA, random_state=seed).fit(emb.loc[fit_rows].to_numpy())
        use = np.concatenate([fit_rows, f.rows["test"]])
        z = pd.DataFrame(pca.transform(emb.loc[use].to_numpy()), index=use,
                         columns=[f"pann_pc{i}" for i in range(N_PCA)])
        feats = z.join(eng, how="left")
        sets = {"PANN (256)": list(z.columns), "PANN (256) + engineered (144)": list(z.columns) + list(eng.columns),
                "engineered (144)": list(eng.columns)}
        for set_name, cols in sets.items():
            for model in S.MODELS:
                probs, setting, val, _ = S.fit_and_test(f, feats, cols, model, seed)
                out.append({"experiment": "C", "features": set_name, "model": model, "fold": k, "setting": setting,
                            "val_balanced_accuracy": val,
                            "pca_variance": float(pca.explained_variance_ratio_.sum()),
                            **M.scores(f.y["test"], probs, f.ships["test"], names)})
                log(f"C fold {k}, {set_name}, {model}: {out[-1]['balanced_accuracy']:.3f} per clip, "
                    f"{out[-1]['per_ship_balanced_accuracy']:.3f} per ship")
    return out


def run_all(data_dir=DATA_DIR, out_dir=OUT_DIR, seed=0, experiments="ABC"):
    with threadpool_limits(4):
        t0 = time.time()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def log(msg):
            print(f"[{(time.time() - t0) / 60:5.1f} min] {msg}", flush=True)

        table = pd.read_parquet(Path(data_dir) / "clips.parquet")
        folds = all_folds(table, seed)
        eng = load_engineered(data_dir)
        results = []
        if "A" in experiments:
            results += experiment_a(table, folds, eng, seed, log)
        if "B" in experiments:
            results += experiment_b(table, folds, eng, seed, log)
        if "C" in experiments:
            results += experiment_c(table, folds, eng, P.load_embeddings(data_dir), seed, log)
        summary = {"left_out": LEFT_OUT, "groupings_a": GROUPINGS_A, "keep_known": KEEP_KNOWN, "n_pca": N_PCA,
                   "minutes": round((time.time() - t0) / 60, 1), "results": results}
        (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
        return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--experiments", default="ABC")
    a = p.parse_args(argv)
    with keep_awake():
        run_all(a.data_dir, a.out_dir, experiments=a.experiments)


if __name__ == "__main__":
    main()
