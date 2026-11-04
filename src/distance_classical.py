"""Distance from the sound, classical models only (lesson 18).

Ship clips of the six-class set (background has no distance), lesson 13's folds and test ships, lesson 13's
engineered features, gradient boosting only:
  - one classifier for 250 m buckets (0-250, ..., 1750-2000 m), scored as: right bucket, right or next bucket
    ("near" counts as correct), and 500 m buckets (pairs of 250 m buckets added together);
  - one regression model for the distance in metres, also turned into buckets for comparison.
Every bucket counts the same in training (weights), and inside a bucket every ship the same.
Which features: all engineered numbers, or with the MFCCs cut to 1-8, decided on the validation folds.

  python src/distance_classical.py      # -> runs/distance_classical/ (CPU, about 15 minutes)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import six_classes as S
from keepawake import keep_awake
from paths import DATA_DIR, RUNS_DIR

OUT_DIR = RUNS_DIR / "distance_classical"
WIDTH, MAX_M = 250, 2000
N_BUCKETS = MAX_M // WIDTH                      # 8
BUCKETS = [f"{a}-{a + WIDTH} m" for a in range(0, MAX_M, WIDTH)]
REDUCED_MFCC = 8
MAX_TREES = 1000                                 # upper limit; early stopping on validation decides
GB = dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0)


def set_width(width: int) -> None:
    """Use buckets of another width (the last bucket ends at 2,000 m or later)."""
    global WIDTH, N_BUCKETS, BUCKETS
    WIDTH, N_BUCKETS = width, int(np.ceil(MAX_M / width))
    BUCKETS = [f"{a}-{a + width} m" for a in range(0, N_BUCKETS * width, width)]


def bucket_of(metres) -> np.ndarray:
    return np.clip(np.asarray(metres) // WIDTH, 0, N_BUCKETS - 1).astype(int)


def bucket_scores(true_b, pred_b) -> dict:
    """Right bucket, right-or-next bucket, and right 500 m bucket (balanced = every true bucket counts the same)."""
    true_b, pred_b = np.asarray(true_b), np.asarray(pred_b)
    near = np.abs(true_b - pred_b) <= 1
    per_bucket_near = [near[true_b == b].mean() for b in np.unique(true_b)]
    out = {"exact": float((true_b == pred_b).mean()), "exact_balanced": balanced_accuracy_score(true_b, pred_b),
           "near": float(near.mean()), "near_balanced": float(np.mean(per_bucket_near))}
    if WIDTH == 250:                                         # pairs of 250 m buckets = 500 m buckets
        out |= {"exact_500m": float((true_b // 2 == pred_b // 2).mean()),
                "exact_500m_balanced": balanced_accuracy_score(true_b // 2, pred_b // 2)}
    return out


def feature_sets(columns) -> dict[str, list[str]]:
    groups = S.feature_groups(columns)
    other = [c for g, cols in groups.items() if g != "MFCC" for c in cols]
    return {"all engineered": S.mfcc_columns(columns, S.N_MFCC) + other,
            f"MFCC 1-{REDUCED_MFCC} + the rest": S.mfcc_columns(columns, REDUCED_MFCC) + other}


def weights(buckets, ships) -> np.ndarray:
    return S.ship_weights(buckets, ships)        # every bucket the same total, every ship the same within it


def fit_clf(X, y, w, Xv, yv, wv, seed=0, n_iter=None):
    m = HistGradientBoostingClassifier(**GB, max_iter=n_iter or MAX_TREES, early_stopping=n_iter is None,
                                       n_iter_no_change=30, random_state=seed)
    return m.fit(X, y, sample_weight=w, **({} if n_iter else dict(X_val=Xv, y_val=yv, sample_weight_val=wv)))


def fit_reg(X, y, w, Xv, yv, wv, seed=0, n_iter=None):
    m = HistGradientBoostingRegressor(**GB, loss="absolute_error", max_iter=n_iter or MAX_TREES,
                                      early_stopping=n_iter is None, n_iter_no_change=30, random_state=seed)
    return m.fit(X, y, sample_weight=w, **({} if n_iter else dict(X_val=Xv, y_val=yv, sample_weight_val=wv)))


def run_all(data_dir=DATA_DIR, out_dir=OUT_DIR, seed=0):
    with threadpool_limits(4):
        return _run_all(data_dir, out_dir, seed)


def _run_all(data_dir, out_dir, seed):
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table, _, eng = S.load_features(data_dir)
    folds = S.make_folds(table, seed=seed)
    ok = (table.label.isin(S.CLASSES6) & (table.label != "background") & table.mean_distance.notna()).to_numpy()
    folds = np.where(ok, folds, 0)                              # ship clips with a known distance only
    dist = table.mean_distance.to_numpy(float)
    ships = S.ship_key(table).to_numpy()
    sets = feature_sets(eng.columns)

    def split(k):
        r = S.split_rows(folds, k)
        r["train+validation"] = np.sort(np.concatenate([r["train"], r["validation"]]))
        return r

    def data(rows, cols):
        b = bucket_of(dist[rows])
        return eng.loc[rows, cols].to_numpy(), b, dist[rows], weights(b, ships[rows])

    # 1. which features: mean validation score over the 5 runs (classifier: balanced exact bucket; regression: MAE)
    choice = []
    for name, cols in sets.items():
        for k in range(1, S.N_FOLDS + 1):
            r = split(k)
            (Xt, bt, dt, wt), (Xv, bv, dv, wv) = data(r["train"], cols), data(r["validation"], cols)
            clf, reg = fit_clf(Xt, bt, wt, Xv, bv, wv, seed), fit_reg(Xt, dt, wt, Xv, dv, wv, seed)
            choice.append({"features": name, "fold": k, "n_features": len(cols),
                           "val_exact_balanced": balanced_accuracy_score(bv, clf.predict(Xv)),
                           "val_mae_m": float(np.average(np.abs(reg.predict(Xv) - dv), weights=wv)),
                           "clf_trees": int(clf.n_iter_), "reg_trees": int(reg.n_iter_)})
        print(f"[{(time.time() - t0) / 60:4.1f} min] validation done: {name}", flush=True)
    choice = pd.DataFrame(choice)
    choice.to_csv(out_dir / "feature_choice.csv", index=False)
    m = choice.groupby("features")[["val_exact_balanced", "val_mae_m", "n_features"]].mean()
    best = m.val_exact_balanced.max()
    ok_sets = m[(m.val_exact_balanced >= best - 0.005)]
    chosen = ok_sets.n_features.idxmin()                       # fewest numbers within half a point of the best
    cols = sets[chosen]
    print(f"chosen: {chosen} ({len(cols)} numbers)\n{m.round(3)}", flush=True)

    # 2. the chosen features on the test folds: refit on train + validation with the trees found on validation
    results, preds = [], []
    for k in range(1, S.N_FOLDS + 1):
        r = split(k)
        c = choice[(choice.features == chosen) & (choice.fold == k)].iloc[0]
        Xa, ba, da, wa = data(r["train+validation"], cols)
        Xs, bs, ds, _ = data(r["test"], cols)
        clf = fit_clf(Xa, ba, wa, None, None, None, seed, n_iter=int(c.clf_trees))
        reg = fit_reg(Xa, da, wa, None, None, None, seed, n_iter=int(c.reg_trees))
        probs, metres = clf.predict_proba(Xs), reg.predict(Xs)
        full = np.zeros((len(Xs), N_BUCKETS)); full[:, clf.classes_] = probs
        pb, rb = full.argmax(1), bucket_of(metres)
        err = np.abs(metres - ds)
        results.append({"fold": k, "features": chosen, "n_features": len(cols),
                        "classifier": bucket_scores(bs, pb), "regression_binned": bucket_scores(bs, rb),
                        "regression": {"mae_m": float(err.mean()), "median_error_m": float(np.median(err)),
                                       "within_250m": float((err <= 250).mean()),
                                       "r2": float(1 - ((metres - ds) ** 2).sum() / ((ds - ds.mean()) ** 2).sum())}})
        preds.append(pd.DataFrame(full, columns=[f"p_{b}" for b in BUCKETS]).assign(
            row=r["test"], fold=k, true_m=ds, true_bucket=bs, clf_bucket=pb, reg_m=metres, reg_bucket=rb,
            label=table.label.to_numpy()[r["test"]], ship=ships[r["test"]]))
        print(f"[{(time.time() - t0) / 60:4.1f} min] fold {k}: classifier exact {results[-1]['classifier']['exact']:.3f}"
              f" near {results[-1]['classifier']['near']:.3f}, regression MAE {err.mean():.0f} m", flush=True)
    pd.concat(preds, ignore_index=True).to_parquet(out_dir / "predictions.parquet")
    summary = {"width_m": WIDTH, "buckets": BUCKETS, "chosen_features": chosen, "n_features": len(cols), "seed": seed,
               "minutes": round((time.time() - t0) / 60, 1), "results": results}
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-dir", type=Path, help="default: runs/distance_classical (250 m) or ..._<width>m")
    p.add_argument("--width", type=int, default=250, help="bucket width in metres")
    a = p.parse_args(argv)
    set_width(a.width)
    out = a.out_dir or (OUT_DIR if a.width == 250 else RUNS_DIR / f"distance_classical_{a.width}m")
    with keep_awake():
        run_all(a.data_dir, out)


if __name__ == "__main__":
    main()
