"""Improving the weakest class, fishing boats (lesson 19).

Lesson 16's five classes (cargo + tanker merged), engineered features (all 144), same folds, test ships and weights.
Fishing boats were right only 43 % of the time per clip, mostly called tug or pleasure craft. Two changes, both
decided on the validation folds only:
  1. the size of the gradient-boosting trees (leaves per tree, clips per leaf), fixed since lesson 13;
  2. one offset per class added to the model's log-probabilities before choosing the class, chosen to give the best
     balanced accuracy on the validation clips (it moves the borderline towards classes the model under-calls).

  python src/fishing_boost.py      # -> runs/fishing_boost/ (CPU, about 30 minutes)
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import five_classes as F5
import six_classes as S
from keepawake import keep_awake
from paths import DATA_DIR, RUNS_DIR

OUT_DIR = RUNS_DIR / "fishing_boost"
LEAVES = (15, 31, 63)
MIN_LEAF = (20, 50)
OFFSET_GRID = np.round(np.arange(-1.0, 1.01, 0.1), 2)
BASE = (15, 50)                                  # lesson 13's setting


def gb(leaves, min_leaf, seed=0, n_iter=None):
    return HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=leaves, min_samples_leaf=min_leaf,
                                          l2_regularization=1.0, max_iter=n_iter or 1000,
                                          early_stopping=n_iter is None, n_iter_no_change=30, random_state=seed)


def best_offsets(probs, y, grid=OFFSET_GRID, passes=3) -> np.ndarray:
    """Per-class additive offsets on log-probabilities that maximise balanced accuracy (coordinate search).
    The first class stays at 0 (only differences between offsets matter)."""
    logp = np.log(np.clip(probs, 1e-9, 1))
    off = np.zeros(probs.shape[1])
    score = balanced_accuracy_score(y, (logp + off).argmax(1))
    for _ in range(passes):
        for c in range(1, probs.shape[1]):
            for v in grid:
                trial = off.copy(); trial[c] = v
                s = balanced_accuracy_score(y, (logp + trial).argmax(1))
                if s > score + 1e-9:
                    score, off = s, trial
    return off


def apply_offsets(probs, off) -> np.ndarray:
    z = np.log(np.clip(probs, 1e-9, 1)) + off
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def run_all(data_dir=DATA_DIR, out_dir=OUT_DIR, seed=0, leaves=LEAVES, min_leaf=MIN_LEAF):
    with threadpool_limits(4):
        return _run_all(data_dir, out_dir, seed, leaves, min_leaf)


def _run_all(data_dir, out_dir, seed, leaves, min_leaf):
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table, _, eng = S.load_features(data_dir)
    cols = list(eng.columns)
    folds = S.make_folds(table, seed=seed)
    fl = [F5.Fold5(table, folds, k) for k in range(1, S.N_FOLDS + 1)]

    # 1. tree size, on validation (models trained on the training folds)
    search, val_probs = [], {}
    for lv, ml in sorted(set(itertools.product(leaves, min_leaf)) | {BASE}):   # lesson 16's setting always in
        for f in fl:
            m = gb(lv, ml, seed).fit(f.X(eng, cols, "train"), f.y["train"], sample_weight=f.w["train"],
                                    X_val=f.X(eng, cols, "validation"), y_val=f.y["validation"],
                                    sample_weight_val=f.w["validation"])
            pv = m.predict_proba(f.X(eng, cols, "validation"))
            val_probs[(lv, ml, f.k)] = pv
            search.append({"leaves": lv, "min_leaf": ml, "fold": f.k, "trees": int(m.n_iter_),
                           "val_balanced_accuracy": balanced_accuracy_score(f.y["validation"], pv.argmax(1))})
        print(f"[{(time.time() - t0) / 60:4.1f} min] leaves {lv}, min leaf {ml}: "
              f"{np.mean([s['val_balanced_accuracy'] for s in search[-len(fl):]]):.3f}", flush=True)
    search = pd.DataFrame(search)
    search.to_csv(out_dir / "tree_search.csv", index=False)
    mean = search.groupby(["leaves", "min_leaf"]).val_balanced_accuracy.mean()
    chosen = tuple(int(v) for v in mean.idxmax())

    # 2. offsets per class, on the validation predictions of the chosen setting (one set per fold)
    offsets = {f.k: best_offsets(val_probs[(*chosen, f.k)], f.y["validation"]) for f in fl}

    # 3. test: refit on train + validation with the trees found on validation; lesson 13's setting for reference
    results, preds = [], []
    for f in fl:
        Xa, Xt = f.X(eng, cols, "train+validation"), f.X(eng, cols, "test")
        for name, setting in [("lesson 16 setting", BASE), ("tuned trees", chosen)]:
            trees = int(search[(search.leaves == setting[0]) & (search.min_leaf == setting[1])
                               & (search.fold == f.k)].trees.iloc[0])
            probs = gb(*setting, seed, n_iter=trees).fit(Xa, f.y["train+validation"],
                                                         sample_weight=f.w["train+validation"]).predict_proba(Xt)
            versions = {name: probs}
            if name == "tuned trees":
                versions["tuned trees + class offsets"] = apply_offsets(probs, offsets[f.k])
            for v, p in versions.items():
                results.append({"fold": f.k, "version": v, "leaves": setting[0], "min_leaf": setting[1],
                                "trees": trees, **F5.scores5(f.y["test"], p, f.ships["test"])})
                preds.append(pd.DataFrame(p, columns=[f"p_{c}" for c in F5.CLASSES5]).assign(
                    row=f.rows["test"], fold=f.k, version=v, y=f.y["test"], ship=f.ships["test"]))
        r = results[-1]
        print(f"[{(time.time() - t0) / 60:4.1f} min] fold {f.k}: {r['balanced_accuracy']:.3f} per clip, "
              f"{r['per_ship_balanced_accuracy']:.3f} per ship, fishing {r['per_class_recall']['fishing']:.3f}",
              flush=True)
    pd.concat(preds, ignore_index=True).to_parquet(out_dir / "predictions.parquet")
    summary = {"classes": F5.CLASSES5, "chosen_trees": {"leaves": chosen[0], "min_leaf": chosen[1]},
               "offsets": {str(k): dict(zip(F5.CLASSES5, map(float, v))) for k, v in offsets.items()},
               "minutes": round((time.time() - t0) / 60, 1), "results": results}
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = p.parse_args(argv)
    with keep_awake():
        s = run_all(a.data_dir, a.out_dir)
    df = pd.DataFrame(s["results"])
    print(df.groupby("version")[["balanced_accuracy", "per_ship_balanced_accuracy"]].mean().round(3))


if __name__ == "__main__":
    main()
