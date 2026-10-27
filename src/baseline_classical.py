"""E1: the "is deep learning worth it?" baseline (lesson 03).

Each clip becomes 257 hand-made numbers (average and spread of each mel band, plus overall loudness);
gradient-boosted decision trees then learn class, distance and speed from those numbers. CPU only.
--features rich adds MFCCs, percentiles and how fast the MFCCs change (761 numbers, ideas from a related
project on the same kind of data).

  python src/baseline_classical.py --fold 1
  python src/baseline_classical.py --fold 1 --features rich     # -> runs/baseline_rich_f1
  python src/baseline_classical.py --fold 1 --merge-into-other pilotvessel dredger rescue   # -> runs/baseline_merged_f1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.fft import dct
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D
import metrics as M
from keepawake import keep_awake
from labels import CLASSES, MERGEABLE
from paths import DATA_DIR, RUNS_DIR

N_MFCC = 20
PERCENTILES = (10, 50, 90)


def band_stats(mel_block: np.ndarray) -> np.ndarray:
    x = np.asarray(mel_block, dtype=np.float32)
    return np.concatenate([x.mean(axis=2), x.std(axis=2), x.mean(axis=(1, 2))[:, None]], axis=1)


def rich_stats(mel_block: np.ndarray) -> np.ndarray:
    """band_stats, plus 20 MFCCs (the DCT of the log-mel bands: the shape of the spectrum in a few numbers) as
    mean and std, the 10th / 50th / 90th percentile of every mel band and MFCC (quiet background, typical level,
    loudest moments), and how fast each MFCC changes (std of its frame-to-frame change)."""
    x = np.asarray(mel_block, dtype=np.float32)
    mfcc = dct(x, type=2, norm="ortho", axis=1)[:, :N_MFCC]
    return np.concatenate([band_stats(x), mfcc.mean(axis=2), mfcc.std(axis=2),
                           *np.percentile(x, PERCENTILES, axis=2), *np.percentile(mfcc, PERCENTILES, axis=2),
                           np.diff(mfcc, axis=2).std(axis=2)], axis=1)


FEATURE_SETS = {"basic": (band_stats, "per-band mean + std + overall mean (257)"),
                "rich": (rich_stats, "basic + MFCC mean/std + mel and MFCC 10/50/90 percentiles + MFCC change (761)")}


def features_for(mel, rows, chunk=250, stats=band_stats):
    # 250 clips at a time: rich_stats needs about 4 copies of a chunk (~250 MB), fine on a busy 16 GB laptop
    return np.concatenate([stats(mel[rows[s:s + chunk]]) for s in range(0, len(rows), chunk)])


def run_baseline(data_dir, fold, out_dir, max_train=20000, seed=0, features="basic", merge=()) -> dict:
    t0 = time.time()
    mel, table = D.open_features(data_dir)
    rng = np.random.default_rng(seed)
    stats, description = FEATURE_SETS[features]
    result = {"fold": fold, "features": description}
    if merge:
        table = D.merge_into_other(table, merge)
        result["merge_into_other"] = list(merge)
    classes = [c for c in CLASSES if c not in merge]
    index = {c: i for i, c in enumerate(classes)}

    for task in ("class", "distance", "speed"):
        tr = D.select_rows(table, fold=fold, split="train", task=task)
        te = D.select_rows(table, fold=fold, split="test", task=task)
        if len(tr) > max_train:
            tr = np.sort(rng.choice(tr, max_train, replace=False))
        Xtr, Xte = features_for(mel, tr, stats=stats), features_for(mel, te, stats=stats)
        if task == "class":
            ytr = np.array([index[c] for c in table.label.iloc[tr]])
            yte = np.array([index[c] for c in table.label.iloc[te]])
            clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed).fit(Xtr, ytr)
            p = clf.predict(Xte)
            result["class"] = {"accuracy": M.accuracy(yte, p), "macro_f1": M.macro_f1(yte, p, len(classes)),
                               "per_class_recall": {c: (None if np.isnan(r) else float(r)) for c, r in
                                                    zip(classes, M.per_class_recall(yte, p, len(classes)))},
                               "n_train": int(len(tr)), "n_test": int(len(te))}
        else:
            col = "mean_distance" if task == "distance" else "mean_sog"
            ytr, yte = table[col].to_numpy(float)[tr], table[col].to_numpy(float)[te]
            reg = HistGradientBoostingRegressor(loss="absolute_error", random_state=seed).fit(Xtr, ytr)
            bands = [700, 1400] if task == "distance" else [6, 12]
            pred = reg.predict(Xte)
            rep = M.regression_report(yte, pred, bands)
            rep.update(unit="m" if task == "distance" else "knots", n_train=int(len(tr)), n_test=int(len(te)),
                       baseline_mae=float(np.mean(np.abs(yte - np.median(ytr)))))
            if task == "distance":
                rep.update(M.distance_bin_report(yte, pred))
                rep["baseline_bins"] = M.distance_bin_report(yte, np.full(len(yte), np.median(ytr)))
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
    p.add_argument("--features", default="basic", choices=list(FEATURE_SETS))
    p.add_argument("--merge-into-other", nargs="+", default=[], metavar="CLASS", choices=MERGEABLE,
                   help="score these classes as 'other' (e.g. pilotvessel dredger rescue)")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    a = p.parse_args(argv)
    with keep_awake():
        try:
            name = ("baseline" + ("" if a.features == "basic" else f"_{a.features}")
                    + ("_merged" if a.merge_into_other else "") + f"_f{a.fold}")
            res = run_baseline(a.data_dir, a.fold, a.runs_dir / name, a.max_train, features=a.features,
                               merge=a.merge_into_other)
        except D.DataError as e:
            sys.exit(f"error: {e}")
    c = res["class"]
    print(f"class: accuracy {c['accuracy']:.3f}, macro-F1 {c['macro_f1']:.3f}")
    print(f"distance: MAE {res['distance']['mae']:.0f} m (always-median guess: {res['distance']['baseline_mae']:.0f} m)")
    print(f"speed: MAE {res['speed']['mae']:.2f} knots (always-median guess: {res['speed']['baseline_mae']:.2f})")


if __name__ == "__main__":
    main()
