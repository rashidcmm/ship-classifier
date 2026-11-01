import json

import numpy as np
import pandas as pd
import pytest

import six_classes as S


def make_table(n_ships=10, clips=(1, 3), seed=0):
    """Six classes, n_ships ships each (background: days), a random number of clips per ship."""
    rng = np.random.default_rng(seed)
    rows, mmsi = [], 100_000_000
    for label in S.CLASSES6:
        for s in range(n_ships):
            mmsi += 1
            for _ in range(rng.integers(clips[0], clips[1] + 1)):
                rows.append({"label": label, "MMSI": 0.0 if label == "background" else float(mmsi),
                             "date": 20220101 + s})
    rows.append({"label": "passengership", "MMSI": 999.0, "date": 20220101})     # a class that is left out
    return pd.DataFrame(rows)


def test_folds_keep_ships_together_and_share_every_class():
    table = make_table()
    folds = S.make_folds(table)
    assert folds[table.label == "passengership"].tolist() == [0]
    sub = table.assign(fold=folds, ship=S.ship_key(table))[folds > 0]
    assert (sub.groupby("ship").fold.nunique() == 1).all()                 # no ship in two folds
    ships = sub.groupby(["label", "fold"]).ship.nunique().unstack()
    assert ships.shape == (6, S.N_FOLDS) and ships.min().min() >= 1


def test_split_rows_are_disjoint_and_complete():
    folds = np.array([0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    r = S.split_rows(folds, 5)
    assert folds[r["test"]].tolist() == [5, 5] and folds[r["validation"]].tolist() == [1, 1]
    together = np.concatenate(list(r.values()))
    assert sorted(together) == list(range(1, 11))


def test_ship_weights_equal_per_class_and_per_ship():
    labels = ["a"] * 4 + ["b"] * 6
    ships = ["s1", "s1", "s1", "s2", "t1", "t2", "t3", "t3", "t3", "t3"]
    w = S.ship_weights(labels, ships)
    assert w.mean() == pytest.approx(1.0)
    assert w[:4].sum() == pytest.approx(w[4:].sum())                       # class a = class b
    assert w[:3].sum() == pytest.approx(w[3])                               # ship s1 (3 clips) = ship s2 (1 clip)
    assert w[4] == pytest.approx(w[5]) == pytest.approx(w[6:].sum())


def test_engineered_features_find_a_tone_and_a_propeller_rate():
    rng = np.random.default_rng(0)
    t = np.arange(S.SR * 5) / S.SR
    noise = rng.normal(size=t.size) * (1 + 0.8 * np.sin(2 * np.pi * 8 * t))  # noise beating 8 times a second
    f = S.engineered_features((0.05 * np.sin(2 * np.pi * 120 * t) + 0.02 * noise).astype(np.float32))
    assert all(np.isfinite(list(f.values())))
    assert f["tonal_count"] >= 1 and f["tonal_strongest_hz"] == pytest.approx(120, abs=3)
    assert f["demon_peak_hz"] == pytest.approx(8, abs=0.3)
    groups = S.feature_groups(f)
    assert sum(len(c) for c in groups.values()) == len(f)                  # every feature is in one group
    assert len(groups["MFCC"]) == 3 * S.N_MFCC and len(S.mfcc_columns(f, 12)) == 36


def test_per_ship_accuracy_averages_the_clips_of_a_ship():
    probs = np.array([[0.6, 0.4], [0.1, 0.9], [0.2, 0.8], [0.9, 0.1]])
    y = np.array([0, 0, 1, 1])
    ships = ["a", "a", "b", "c"]          # ship a: mean (0.35, 0.65) -> wrong; b right; c wrong
    assert S.per_ship_balanced_accuracy(probs, y, ships, n_classes=2) == pytest.approx((0 + 0.5) / 2)


def test_run_all_on_synthetic_features(tmp_path):
    rng = np.random.default_rng(0)
    table = make_table(n_ships=8, clips=(2, 2))
    table.to_parquet(tmp_path / "clips.parquet")
    rows = np.flatnonzero(table.label.isin(S.CLASSES6))
    y = table.label.iloc[rows].map({c: i for i, c in enumerate(S.CLASSES6)}).to_numpy()
    eng_names = list(S.engineered_features(rng.normal(size=S.SR * 5).astype(np.float32)))
    for name, cols in [(S.SIMPLE_FILE, [f"m{i}" for i in range(20)]), (S.ENGINEERED_FILE, eng_names)]:
        X = rng.normal(size=(len(rows), len(cols)))
        X[:, :6] += 3 * np.eye(6)[y]                                       # the first columns carry the class
        pd.DataFrame(X, columns=cols).assign(row=rows).to_parquet(tmp_path / name)
    summary = S.run_all(data_dir=tmp_path, out_dir=tmp_path / "out")
    saved = json.loads((tmp_path / "out" / "results.json").read_text())
    assert saved["n_mfcc"] in S.MFCC_CHOICES and len(saved["results"]) == S.N_FOLDS * 4
    simple = [r["balanced_accuracy"] for r in summary["results"] if r["features"] == "simple (no MFCC)"]
    assert np.mean(simple) > 0.5                                            # chance is 1/6
    for f in ("predictions.parquet", "folds.parquet", "mfcc_sweep.csv", "group_importance.csv", "group_alone.csv"):
        assert (tmp_path / "out" / f).exists()
