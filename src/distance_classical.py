"""Distance from the sound, classical models only (lesson 18).

Ship clips of the six-class set (background has no distance), lesson 13's folds and test ships, lesson 13's
engineered features, gradient boosting only:
  - one classifier for 250 m buckets (0-250, ..., 1750-2000 m), scored as: right bucket, right or next bucket
    ("near" counts as correct), and 500 m buckets (pairs of 250 m buckets added together);
  - one regression model for the distance in metres, also turned into buckets for comparison.
Every bucket counts the same in training (weights), and inside a bucket every ship the same.
Which features: all engineered numbers, or with the MFCCs cut to 1-8, decided on the validation folds.

  python src/distance_classical.py      # -> runs/distance_classical/ (CPU, about 15 minutes)
  python src/distance_classical.py --ship-type   # + the TRUE ship type as 5 on/off inputs (an upper bound:
                                                 #   how much would knowing the ship type help?) -> ..._shiptype
  python src/distance_classical.py --size        # + ship length, beam, draught (AIS) and which hydrophone: how much
                                                 #   would knowing how big (how loud) the ship is help? -> ..._size
  python src/distance_classical.py --model svm   # SVM classifier + SVR (RBF kernel, standardised features, C in
                                                 #   {1, 10} on validation), on the features gradient boosting chose
                                                 #   -> ..._svm (slow: about an hour per target)
  python src/distance_classical.py --model linear  # logistic regression + ridge regression (standardised; C / alpha
                                                   #   on validation), same features as gradient boosting -> ..._linear
  python src/distance_classical.py --neighbours  # + features from the same ship's other clips in the same passage
                                                 #   (level trend, level below the loudest clip, tone change) -> ..._nb
  python src/distance_classical.py --speed-input # distance only: + the TRUE speed (upper bound) -> ..._speed
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import six_classes as S
from keepawake import keep_awake
from paths import DATA_DIR, RUNS_DIR

OUT_DIR = RUNS_DIR / "distance_classical"
WIDTH, MAX_M = 250, 2000
TARGET, UNIT = "mean_distance", "m"            # --target speed: "mean_sog", "kn" (speed over ground, knots)
N_BUCKETS = MAX_M // WIDTH                      # 8
BUCKETS = [f"{a}-{a + WIDTH} m" for a in range(0, MAX_M, WIDTH)]
REDUCED_MFCC = 8
MAX_TREES = 1000                                 # upper limit; early stopping on validation decides
GB = dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0)
SHIP_TYPE = False                                # --ship-type: add the true ship type (one-hot) to every feature set
SIZE = False                                     # --size: add length, beam, draught (AIS; draught may be missing) + hydrophone
SIZE_COLS = ["length", "beam", "draught"]
MODEL = "gb"                                     # --model svm: SVC + SVR; --model linear: logistic + ridge
SVM_C = (1.0, 10.0)
LIN_C, RIDGE_ALPHA = (0.01, 0.1, 1.0), (1.0, 10.0, 100.0)
NEIGHBOURS = False                               # --neighbours: add the passage features (neighbour_features)
SPEED_INPUT = False                              # --speed-input: add the true speed (distance target only)
PASSAGE_GAP_S = 300                              # clips of one ship on one hydrophone at most 5 min apart = one passage
SVM_FEATURES = None                              # the feature set gradient boosting chose (read from its results.json)


def set_width(width: int) -> None:
    """Use buckets of another width (the last bucket ends at 2,000 m or later)."""
    global WIDTH, N_BUCKETS, BUCKETS
    WIDTH, N_BUCKETS = width, int(np.ceil(MAX_M / width))
    BUCKETS = [f"{a}-{a + width} {UNIT}" for a in range(0, N_BUCKETS * width, width)]
    if TARGET == "mean_sog":                                  # the last speed bucket is open-ended
        BUCKETS[-1] = f"{(N_BUCKETS - 1) * width}+ {UNIT}"


def set_target(target: str) -> None:
    """'distance' (metres, up to 2,000 m) or 'speed' (knots; buckets up to 16 kn, the last one open)."""
    global TARGET, UNIT, MAX_M
    TARGET, UNIT, MAX_M = ("mean_distance", "m", 2000) if target == "distance" else ("mean_sog", "kn", 20)


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


def svm_clf(C, X, y, w):
    return make_pipeline(StandardScaler(), SVC(C=C, kernel="rbf", gamma="scale")).fit(X, y, svc__sample_weight=w)


def svm_reg(C, X, y, w):
    """SVR on standardised features and a standardised target (epsilon-insensitive: close to absolute error)."""
    m = TransformedTargetRegressor(make_pipeline(StandardScaler(), SVR(C=C, kernel="rbf", gamma="scale")),
                                   transformer=StandardScaler())
    return m.fit(X, y, svr__sample_weight=w)


def lin_clf(C, X, y, w):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(
        X, y, logisticregression__sample_weight=w)


def lin_reg(alpha, X, y, w):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X, y, ridge__sample_weight=w)


def neighbour_features(table: pd.DataFrame, eng: pd.DataFrame) -> pd.DataFrame:
    """Features from the same ship's other clips in the same passage (same MMSI and hydrophone, clips at most
    PASSAGE_GAP_S apart). Only audio features and clip times are used, never distance or speed, so they are
    available for a new recording too. NaN where a clip has no neighbour."""
    rows = eng.index.to_numpy()
    start = table.path.str.extract(r"(\d{8}T\d{6})")[0].to_numpy()[rows]
    t = pd.DataFrame({
        "ship": table.MMSI.to_numpy()[rows], "hyd": table.hydrophone.astype(str).to_numpy()[rows],
        "start": pd.to_datetime(start, format="%Y%m%dT%H%M%S"),
        "level": eng.level_broadband.to_numpy(), "tone": eng.tonal_strongest_hz.replace(0, np.nan).to_numpy()},
        index=rows)
    t = t[pd.notna(t.ship)].sort_values(["ship", "hyd", "start"])
    sec = (t.start - t.start.min()).dt.total_seconds().to_numpy()
    same = (t.ship.to_numpy()[1:] == t.ship.to_numpy()[:-1]) & (t.hyd.to_numpy()[1:] == t.hyd.to_numpy()[:-1])
    new = np.r_[True, ~(same & (np.diff(sec) <= PASSAGE_GAP_S))]
    t["passage"], t["sec"] = np.cumsum(new), sec
    g = t.groupby("passage")
    out = pd.DataFrame(index=t.index)
    out["nb_clips"] = g["level"].transform("size").astype(float)
    out["nb_passage_s"] = g["sec"].transform("max") - g["sec"].transform("min")
    out["nb_position"] = (t.sec - g["sec"].transform("min")) / out.nb_passage_s.replace(0, np.nan)
    out["nb_level_below_max"] = g["level"].transform("max") - t.level
    out["nb_level_minus_mean"] = t.level - g["level"].transform("mean")
    for col, name in (("level", "nb_level_db_per_min"), ("tone", "nb_tone_hz_per_min")):
        # previous and next clip of the passage; at either end the clip itself stands in (one-sided slope)
        v0, t0 = g[col].shift(1).fillna(t[col]), g["sec"].shift(1).fillna(t.sec)
        v1, t1 = g[col].shift(-1).fillna(t[col]), g["sec"].shift(-1).fillna(t.sec)
        out[name] = (v1 - v0) / ((t1 - t0) / 60).replace(0, np.nan)
    single = out.nb_clips <= 1
    out.loc[single, [c for c in out.columns if c != "nb_clips"]] = np.nan
    return out.reindex(rows)


def run_all(data_dir=DATA_DIR, out_dir=OUT_DIR, seed=0):
    with threadpool_limits(4):
        return _run_all(data_dir, out_dir, seed)


def _run_all(data_dir, out_dir, seed):
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table, _, eng = S.load_features(data_dir)
    folds = S.make_folds(table, seed=seed)
    ok = (table.label.isin(S.CLASSES6) & (table.label != "background") & table[TARGET].notna()).to_numpy()
    folds = np.where(ok, folds, 0)                              # ship clips with a known distance only
    dist = table[TARGET].to_numpy(float)
    ships = S.ship_key(table).to_numpy()
    sets = feature_sets(eng.columns)
    if MODEL in ("svm", "linear"):                              # same features as gradient boosting chose
        sets = {SVM_FEATURES: sets[SVM_FEATURES]}
    if NEIGHBOURS:
        nb = neighbour_features(table, eng)
        eng = eng.join(nb)
        sets = {f"{n} + neighbours": cols + list(nb.columns) for n, cols in sets.items()}
    if SPEED_INPUT:
        eng = eng.copy()
        eng["true_speed_kn"] = table.mean_sog.to_numpy(float)[eng.index.to_numpy()]
        sets = {f"{n} + true speed": cols + ["true_speed_kn"] for n, cols in sets.items()}
    if SHIP_TYPE:                                               # true ship type, one on/off column per ship class
        labels = table.label.to_numpy()[eng.index.to_numpy()]  # eng holds only some rows: index = row in table
        type_cols = [f"type_{c}" for c in S.CLASSES6 if c != "background"]
        eng = eng.copy()
        for c in type_cols:
            eng[c] = (labels == c[5:]).astype(float)
        sets = {f"{n} + ship type": cols + type_cols for n, cols in sets.items()}
    if SIZE:                                                    # ship size (AIS) and which hydrophone recorded the clip
        eng = eng.copy()
        size_cols = [f"ais_{c}" for c in SIZE_COLS] + ["hydrophone_2556"]
        for c in SIZE_COLS:
            v = table[c].to_numpy(float)[eng.index.to_numpy()]
            eng[f"ais_{c}"] = np.where(v > 0, v, np.nan)        # 0 in AIS means "not reported"
        eng["hydrophone_2556"] = (table.hydrophone.to_numpy()[eng.index.to_numpy()] == "ICLISTENAF2556").astype(float)
        sets = {f"{n} + size + hydrophone": cols + size_cols for n, cols in sets.items()}

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
            if MODEL in ("svm", "linear"):                      # C / alpha chosen per target on this validation fold
                fc, fr, Cc, Cr = ((svm_clf, svm_reg, SVM_C, SVM_C) if MODEL == "svm" else
                                  (lin_clf, lin_reg, LIN_C, RIDGE_ALPHA))
                ce = {C: balanced_accuracy_score(bv, fc(C, Xt, bt, wt).predict(Xv)) for C in Cc}
                cm = {C: float(np.average(np.abs(fr(C, Xt, dt, wt).predict(Xv) - dv), weights=wv)) for C in Cr}
                cc, rc = max(ce, key=ce.get), min(cm, key=cm.get)
                choice.append({"features": name, "fold": k, "n_features": len(cols), "val_exact_balanced": ce[cc],
                               "val_mae_m": cm[rc], "clf_C": cc, "reg_C": rc, "clf_trees": 0, "reg_trees": 0})
                print(f"[{(time.time() - t0) / 60:4.1f} min] validation fold {k}: classifier C={cc}, regression C={rc}",
                      flush=True)
                continue
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
        if MODEL in ("svm", "linear"):                          # no probabilities kept: one-hot of the predicted bucket
            fc, fr = (svm_clf, svm_reg) if MODEL == "svm" else (lin_clf, lin_reg)
            clf, reg = fc(float(c.clf_C), Xa, ba, wa), fr(float(c.reg_C), Xa, da, wa)
            full = np.eye(N_BUCKETS)[clf.predict(Xs)]
            metres = reg.predict(Xs)
        else:
            clf = fit_clf(Xa, ba, wa, None, None, None, seed, n_iter=int(c.clf_trees))
            reg = fit_reg(Xa, da, wa, None, None, None, seed, n_iter=int(c.reg_trees))
            probs, metres = clf.predict_proba(Xs), reg.predict(Xs)
            full = np.zeros((len(Xs), N_BUCKETS)); full[:, clf.classes_] = probs
        pb, rb = full.argmax(1), bucket_of(metres)
        err = np.abs(metres - ds)
        results.append({"fold": k, "features": chosen, "n_features": len(cols),
                        "classifier": bucket_scores(bs, pb), "regression_binned": bucket_scores(bs, rb),
                        "regression": {"mae_m": float(err.mean()), "median_error_m": float(np.median(err)),
                                       f"within_{WIDTH}{UNIT}": float((err <= WIDTH).mean()),
                                       "r2": float(1 - ((metres - ds) ** 2).sum() / ((ds - ds.mean()) ** 2).sum())}})
        preds.append(pd.DataFrame(full, columns=[f"p_{b}" for b in BUCKETS]).assign(
            row=r["test"], fold=k, true_m=ds, true_bucket=bs, clf_bucket=pb, reg_m=metres, reg_bucket=rb,
            label=table.label.to_numpy()[r["test"]], ship=ships[r["test"]]))
        print(f"[{(time.time() - t0) / 60:4.1f} min] fold {k}: classifier exact {results[-1]['classifier']['exact']:.3f}"
              f" near {results[-1]['classifier']['near']:.3f}, regression MAE {err.mean():.0f} m", flush=True)
    pd.concat(preds, ignore_index=True).to_parquet(out_dir / "predictions.parquet")
    summary = {"model": MODEL, "neighbours": NEIGHBOURS, "speed_input": SPEED_INPUT, "ship_type_input": SHIP_TYPE, "size_input": SIZE, "target": TARGET, "unit": UNIT, "width": WIDTH, "buckets": BUCKETS, "chosen_features": chosen, "n_features": len(cols), "seed": seed,
               "minutes": round((time.time() - t0) / 60, 1), "results": results}
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-dir", type=Path, help="default: runs/distance_classical (250 m) or ..._<width>m")
    p.add_argument("--target", choices=["distance", "speed"], default="distance")
    p.add_argument("--width", type=int, help="bucket width (default 250 m, or 4 knots for speed)")
    p.add_argument("--ship-type", action="store_true", help="add the true ship type as input (upper-bound test)")
    p.add_argument("--size", action="store_true", help="add ship length, beam, draught (AIS) and the hydrophone")
    p.add_argument("--model", choices=["gb", "svm", "linear"], default="gb",
                   help="gradient boosting (default), SVM/SVR, or logistic + ridge regression")
    p.add_argument("--neighbours", action="store_true", help="add features from the same passage's other clips")
    p.add_argument("--speed-input", action="store_true", help="distance only: add the true speed (upper bound)")
    a = p.parse_args(argv)
    global SHIP_TYPE, SIZE, MODEL, SVM_FEATURES, NEIGHBOURS, SPEED_INPUT
    SHIP_TYPE, SIZE, MODEL, NEIGHBOURS, SPEED_INPUT = a.ship_type, a.size, a.model, a.neighbours, a.speed_input
    if MODEL != "gb" and (SHIP_TYPE or SIZE or NEIGHBOURS or SPEED_INPUT):
        p.error("--model svm / linear run on the sound features only")
    if SPEED_INPUT and a.target != "distance":
        p.error("--speed-input is for the distance target")
    set_target(a.target)
    width = a.width or (250 if a.target == "distance" else 4)
    set_width(width)
    out = a.out_dir or (RUNS_DIR / f"speed_classical_{width}kn" if a.target == "speed" else
                        OUT_DIR if width == 250 else RUNS_DIR / f"distance_classical_{width}m")
    if MODEL in ("svm", "linear"):                              # the gradient-boosting run of the same target
        SVM_FEATURES = json.loads((out / "results.json").read_text())["chosen_features"]
    if not a.out_dir:
        out = out.with_name(out.name + ("_shiptype" if a.ship_type else "") + ("_size" if a.size else "")
                            + ("_nb" if NEIGHBOURS else "") + ("_speed" if SPEED_INPUT else "")
                            + ("" if MODEL == "gb" else f"_{MODEL}"))
    with keep_awake():
        run_all(a.data_dir, out)


if __name__ == "__main__":
    main()
