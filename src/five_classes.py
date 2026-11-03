"""Five classes: cargo and tanker merged into one class (lesson 16).

Lesson 13's six classes, but cargo and tanker become one class, "cargo + tanker": their spectra are almost the same
(lesson 15, section 1) and they cause most of the six-class errors (each is called the other 30-44 % of the time).
Everything else stays as in lesson 13: the same folds and test ships, the same training weights (every original
class, and every ship within it, counts the same, so cargo and tanker each make up half of the merged class), the
same two feature sets and the same two models. Both models' probabilities are also averaged ("both averaged").

Scores are reported per clip (every 5-s clip judged on its own) and per ship (the probabilities of all clips of a ship
in the test fold are averaged into one answer; background clips are grouped by recording day).

  python src/five_classes.py      # -> runs/five_classes/ (about 15 minutes, CPU; needs lesson 13's features)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import six_classes as S
from keepawake import keep_awake
from paths import DATA_DIR, RUNS_DIR

CLASSES5 = ["background", "cargo + tanker", "fishing", "pleasurecraft", "tug"]
SIX_TO_FIVE = np.array([CLASSES5.index("cargo + tanker" if c in ("cargo", "tanker") else c) for c in S.CLASSES6])
OUT_DIR = RUNS_DIR / "five_classes"
THREADS = 4                    # see six_classes_fewer.THREADS: the default thread count is very slow on this laptop
BOTH = "both averaged"


def merge_probs(probs6: np.ndarray) -> np.ndarray:
    """Six-class probabilities -> five, by adding the cargo and tanker columns."""
    out = np.zeros((len(probs6), len(CLASSES5)))
    for six, five in enumerate(SIX_TO_FIVE):
        out[:, five] += probs6[:, six]
    return out


def scores5(y, probs, ships) -> dict:
    """Per clip (accuracy, balanced accuracy, macro-F1, recall per class) and per ship (balanced accuracy)."""
    pred = probs.argmax(1)
    recall = confusion_matrix(y, pred, labels=range(len(CLASSES5)), normalize="true").diagonal()
    return {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro"),
            "per_ship_balanced_accuracy": S.per_ship_balanced_accuracy(probs, y, ships),
            "per_class_recall": dict(zip(CLASSES5, map(float, recall)))}


class Fold5(S.Fold):
    """Lesson 13's fold with five-class labels. The weights stay the six-class ones (y6 keeps the original labels)."""

    def __init__(self, table, folds, k):
        super().__init__(table, folds, k)
        self.y6 = self.y
        self.y = {s: SIX_TO_FIVE[y] for s, y in self.y6.items()}


def run_all(data_dir=DATA_DIR, out_dir=OUT_DIR, seed=0):
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table, simple, eng = S.load_features(data_dir)
    folds = S.make_folds(table, seed=seed)
    sets = {"simple (no MFCC)": (simple, list(simple.columns)), "engineered": (eng, list(eng.columns))}
    results, preds = [], []
    for k in range(1, S.N_FOLDS + 1):
        f = Fold5(table, folds, k)
        for name, (feats, cols) in sets.items():
            probs = {}
            for model in S.MODELS:
                probs[model], setting, val, _ = S.fit_and_test(f, feats, cols, model, seed)
                results.append({"fold": k, "features": name, "model": model, "n_features": len(cols),
                                "setting": setting, "val_balanced_accuracy": val,
                                **scores5(f.y["test"], probs[model], f.ships["test"])})
            probs[BOTH] = (probs[S.MODELS[0]] + probs[S.MODELS[1]]) / 2
            results.append({"fold": k, "features": name, "model": BOTH, "n_features": len(cols),
                            **scores5(f.y["test"], probs[BOTH], f.ships["test"])})
            for model, p in probs.items():
                preds.append(pd.DataFrame(p, columns=[f"p_{c}" for c in CLASSES5]).assign(
                    row=f.rows["test"], fold=k, features=name, model=model, y=f.y["test"], y6=f.y6["test"],
                    ship=f.ships["test"]))
            print(f"[{(time.time() - t0) / 60:5.1f} min] fold {k}, {name}: " + ", ".join(
                f"{r['model']} {r['balanced_accuracy']:.3f} / ship {r['per_ship_balanced_accuracy']:.3f}"
                for r in results[-3:]), flush=True)
    pd.concat(preds, ignore_index=True).to_parquet(out_dir / "predictions.parquet")
    summary = {"classes": CLASSES5, "merged": {"cargo + tanker": ["cargo", "tanker"]}, "seed": seed,
               "minutes": round((time.time() - t0) / 60, 1), "results": results}
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--threads", type=int, default=THREADS)
    a = p.parse_args(argv)
    with keep_awake(), threadpool_limits(a.threads):
        summary = run_all(a.data_dir, a.out_dir)
    df = pd.DataFrame(summary["results"])
    print(df.groupby(["features", "model"])[["balanced_accuracy", "per_ship_balanced_accuracy"]].agg(
        ["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
