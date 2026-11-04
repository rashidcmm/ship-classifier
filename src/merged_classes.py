"""Merged classes with properly balanced weights, gradient boosting and SVM (lesson 20).

Lesson 16 merged cargo and tanker but kept the six-class weights, so the merged class counted twice as much as each
other class in training. Here every merged class gets the same total weight; inside it, each original class gets
the same share (cargo half, tanker half), and inside those every ship the same.

Two groupings, lesson 13's folds, test ships and 144 engineered features:
  - 5 classes: background, cargo + tanker, fishing, pleasure craft, tug
  - 4 classes: background, cargo + tanker, fishing + tug ("small working boats"), pleasure craft
Two models: gradient boosting (lesson 13's settings) and an SVM with an RBF kernel (standardised features; C chosen
on the validation fold). Scored per clip and per ship (lesson 16).

  python src/merged_classes.py      # -> runs/merged_classes/ (CPU, about an hour, mostly the SVM)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import six_classes as S
from keepawake import keep_awake
from paths import DATA_DIR, RUNS_DIR

OUT_DIR = RUNS_DIR / "merged_classes"
GROUPINGS = {
    "5 classes": {"background": ["background"], "cargo + tanker": ["cargo", "tanker"], "fishing": ["fishing"],
                  "pleasurecraft": ["pleasurecraft"], "tug": ["tug"]},
    "4 classes": {"background": ["background"], "cargo + tanker": ["cargo", "tanker"],
                  "fishing + tug": ["fishing", "tug"], "pleasurecraft": ["pleasurecraft"]},
}
SVM_C = (1.0, 10.0)
MODELS = ("gradient boosting", "SVM")


def mapping(grouping: dict) -> np.ndarray:
    """Six-class index -> merged-class index."""
    return np.array([next(i for i, members in enumerate(grouping.values()) if c in members) for c in S.CLASSES6])


def merged_weights(y6, ships, grouping: dict) -> np.ndarray:
    """Every merged class the same total; inside it every original class the same; inside that every ship."""
    w = S.ship_weights(y6, ships)                              # every original class the same total
    size = np.array([len(m) for m in grouping.values()])[mapping(grouping)[np.asarray(y6)]]
    w = w / size
    return w / w.mean()


class MergedFold(S.Fold):
    def __init__(self, table, folds, k, grouping):
        super().__init__(table, folds, k)
        to = mapping(grouping)
        self.y6 = self.y
        self.y = {s: to[y] for s, y in self.y6.items()}
        self.w = {s: merged_weights(self.y6[s], self.ships[s], grouping) for s in self.rows}


def scores(y, probs, ships, names) -> dict:
    pred = probs.argmax(1)
    recall = confusion_matrix(y, pred, labels=range(len(names)), normalize="true").diagonal()
    return {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro"),
            "per_ship_balanced_accuracy": S.per_ship_balanced_accuracy(probs, y, ships),
            "per_class_recall": dict(zip(names, map(float, recall)))}


def softmax(z):
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def svm(C):
    return make_pipeline(StandardScaler(), SVC(C=C, kernel="rbf", gamma="scale", decision_function_shape="ovr"))


def fit_and_test_svm(f, feats, cols):
    """C chosen on validation, refit on train + validation; returns (test scores as probabilities, C, val score).
    The SVM gives decision values, not probabilities: a softmax turns them into weights that can be averaged per
    ship (the class with the highest value is the same either way)."""
    Xtr, Xva, Xall, Xte = (f.X(feats, cols, s) for s in ("train", "validation", "train+validation", "test"))
    best = None
    for C in SVM_C:
        m = svm(C).fit(Xtr, f.y["train"], svc__sample_weight=f.w["train"])
        val = balanced_accuracy_score(f.y["validation"], m.predict(Xva))
        if best is None or val > best[1]:
            best = (C, val)
    final = svm(best[0]).fit(Xall, f.y["train+validation"], svc__sample_weight=f.w["train+validation"])
    return softmax(final.decision_function(Xte)), best[0], best[1]


def run_all(data_dir=DATA_DIR, out_dir=OUT_DIR, seed=0, groupings=GROUPINGS, models=MODELS):
    with threadpool_limits(4):
        return _run_all(data_dir, out_dir, seed, groupings, models)


def _run_all(data_dir, out_dir, seed, groupings, models):
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table, _, eng = S.load_features(data_dir)
    cols = list(eng.columns)
    folds = S.make_folds(table, seed=seed)
    results, preds = [], []
    for gname, grouping in groupings.items():
        names = list(grouping)
        for model in models:
            for k in range(1, S.N_FOLDS + 1):
                f = MergedFold(table, folds, k, grouping)
                if model == "SVM":
                    probs, setting, val = fit_and_test_svm(f, eng, cols)
                else:
                    probs, setting, val, _ = S.fit_and_test(f, eng, cols, model, seed)
                results.append({"grouping": gname, "model": model, "fold": k, "setting": setting,
                                "val_balanced_accuracy": val, **scores(f.y["test"], probs, f.ships["test"], names)})
                preds.append(pd.DataFrame(probs, columns=[f"p_{c}" for c in names]).assign(
                    row=f.rows["test"], fold=k, grouping=gname, model=model, y=f.y["test"], y6=f.y6["test"],
                    ship=f.ships["test"]))
                r = results[-1]
                print(f"[{(time.time() - t0) / 60:5.1f} min] {gname}, {model}, fold {k}: "
                      f"{r['balanced_accuracy']:.3f} per clip, {r['per_ship_balanced_accuracy']:.3f} per ship",
                      flush=True)
                (out_dir / "results.json").write_text(json.dumps({"groupings": groupings, "results": results},
                                                                 indent=2))
    pd.concat(preds, ignore_index=True).to_parquet(out_dir / "predictions.parquet")
    summary = {"groupings": groupings, "svm_C": SVM_C, "minutes": round((time.time() - t0) / 60, 1),
               "results": results}
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
    print(df.groupby(["grouping", "model"])[["balanced_accuracy", "per_ship_balanced_accuracy"]].agg(
        ["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
