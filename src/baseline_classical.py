"""E1: the "is deep learning worth it?" baseline (lesson 03).

Each clip becomes 257 hand-made numbers (average and spread of each mel band, plus overall loudness);
gradient-boosted decision trees then learn class, distance and speed from those numbers. CPU only.

  python src/baseline_classical.py --fold 1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D
import metrics as M
from keepawake import keep_awake
from labels import CLASSES
from paths import DATA_DIR, RUNS_DIR


def band_stats(mel_block: np.ndarray) -> np.ndarray:
    x = np.asarray(mel_block, dtype=np.float32)
    return np.concatenate([x.mean(axis=2), x.std(axis=2), x.mean(axis=(1, 2))[:, None]], axis=1)


def features_for(mel, rows, chunk=1000):
    return np.concatenate([band_stats(mel[rows[s:s + chunk]]) for s in range(0, len(rows), chunk)])


def run_baseline(data_dir, fold, out_dir, max_train=20000, seed=0) -> dict:
    t0 = time.time()
    mel, table = D.open_features(data_dir)
    rng = np.random.default_rng(seed)
    result = {"fold": fold, "features": "per-band mean + std + overall mean (257)"}
    index = {c: i for i, c in enumerate(CLASSES)}

    for task in ("class", "distance", "speed"):
        tr = D.select_rows(table, fold=fold, split="train", task=task)
        te = D.select_rows(table, fold=fold, split="test", task=task)
        if len(tr) > max_train:
            tr = np.sort(rng.choice(tr, max_train, replace=False))
        Xtr, Xte = features_for(mel, tr), features_for(mel, te)
        if task == "class":
            ytr = np.array([index[c] for c in table.label.iloc[tr]])
            yte = np.array([index[c] for c in table.label.iloc[te]])
            clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed).fit(Xtr, ytr)
            p = clf.predict(Xte)
            result["class"] = {"accuracy": M.accuracy(yte, p), "macro_f1": M.macro_f1(yte, p, len(CLASSES)),
                               "per_class_recall": {c: (None if np.isnan(r) else float(r)) for c, r in
                                                    zip(CLASSES, M.per_class_recall(yte, p, len(CLASSES)))},
                               "n_train": int(len(tr)), "n_test": int(len(te))}
        else:
            col = "mean_distance" if task == "distance" else "mean_sog"
            ytr, yte = table[col].to_numpy(float)[tr], table[col].to_numpy(float)[te]
            reg = HistGradientBoostingRegressor(loss="absolute_error", random_state=seed).fit(Xtr, ytr)
            bands = [700, 1400] if task == "distance" else [6, 12]
            rep = M.regression_report(yte, reg.predict(Xte), bands)
            rep.update(unit="m" if task == "distance" else "knots", n_train=int(len(tr)), n_test=int(len(te)),
                       baseline_mae=float(np.mean(np.abs(yte - np.median(ytr)))))
            result[task] = rep
    result["seconds"] = round(time.time() - t0, 1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fold", type=int, default=1, choices=range(1, 6))
    p.add_argument("--max-train", type=int, default=20000)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    a = p.parse_args(argv)
    with keep_awake():
        try:
            res = run_baseline(a.data_dir, a.fold, a.runs_dir / f"baseline_f{a.fold}", a.max_train)
        except D.DataError as e:
            sys.exit(f"error: {e}")
    c = res["class"]
    print(f"class: accuracy {c['accuracy']:.3f}, macro-F1 {c['macro_f1']:.3f}")
    print(f"distance: MAE {res['distance']['mae']:.0f} m (always-median guess: {res['distance']['baseline_mae']:.0f} m)")
    print(f"speed: MAE {res['speed']['mae']:.2f} knots (always-median guess: {res['speed']['baseline_mae']:.2f})")


if __name__ == "__main__":
    main()
