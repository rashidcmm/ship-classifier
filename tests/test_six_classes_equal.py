import json

import numpy as np
import pandas as pd

import six_classes as S
import six_classes_equal as E


def test_balance_gives_equal_ships_and_clips_spread_over_ships():
    # class a: 3 ships with 1 clip each; class b: 5 ships with 4 clips each
    labels = ["a"] * 3 + ["b"] * 20
    ships = ["a1", "a2", "a3"] + [f"b{i}" for i in range(5) for _ in range(4)]
    keep = E.balance_by_removal(labels, ships, seed=0)
    kept = pd.DataFrame({"label": np.array(labels)[keep], "ship": np.array(ships)[keep]})
    assert (kept.groupby("label").ship.nunique() == 3).all()                # as many ships as the smallest class
    assert (kept.groupby("label").size() == 3).all()                        # as many clips as the smallest total
    assert (kept.groupby("ship").size() == 1).all()                         # one clip from each kept ship first
    assert np.array_equal(keep, E.balance_by_removal(labels, ships, seed=0))
    assert not np.array_equal(keep, E.balance_by_removal(labels, ships, seed=1))


def test_equal_fold_cuts_training_but_not_test():
    table = pd.DataFrame([{"label": c, "MMSI": float(1000 * i + s), "date": 20220101 + s}
                          for i, c in enumerate(S.CLASSES6) for s in range(5 + 3 * i) for _ in range(1 + i % 3)])
    folds = S.make_folds(table)
    full = S.Fold(table, folds, 1)
    f = E.EqualFold(full, seed=0)
    assert np.array_equal(f.rows["test"], full.rows["test"])
    for split in ("train", "validation", "train+validation"):
        c = f.counts(split)
        assert c.ships.nunique() == 1 and c.clips.nunique() == 1 and len(c) == 6
        assert set(f.rows[split]) <= set(full.rows[split]) and f.w[split] is None


def test_run_equal_on_synthetic_features(tmp_path):
    rng = np.random.default_rng(0)
    table = pd.DataFrame([{"label": c, "MMSI": float(1000 * i + s), "date": 20220101 + s}
                          for i, c in enumerate(S.CLASSES6) for s in range(10 + 4 * i) for _ in range(2)])
    table.to_parquet(tmp_path / "clips.parquet")
    rows = np.arange(len(table))
    y = table.label.map({c: i for i, c in enumerate(S.CLASSES6)}).to_numpy()
    eng_names = list(S.engineered_features(rng.normal(size=S.SR * 5).astype(np.float32)))
    for name, cols in [(S.SIMPLE_FILE, [f"m{i}" for i in range(20)]), (S.ENGINEERED_FILE, eng_names)]:
        X = rng.normal(size=(len(rows), len(cols)))
        X[:, :6] += 3 * np.eye(6)[y]
        pd.DataFrame(X, columns=cols).assign(row=rows).to_parquet(tmp_path / name)
    summary = E.run_equal(data_dir=tmp_path, out_dir=tmp_path / "out", draws=2, n_mfcc=4)
    saved = json.loads((tmp_path / "out" / "results.json").read_text())
    assert len(saved["results"]) == S.N_FOLDS * 2 * 4
    # (gradient boosting can't split 16 clips per class into leaves of 50, so only logistic regression is checked)
    assert np.mean([r["balanced_accuracy"] for r in summary["results"] if r["model"] == "logistic regression"]) > 0.5
    counts = pd.read_csv(tmp_path / "out" / "counts.csv")
    assert (counts.groupby(["fold", "draw", "split"]).clips.nunique() == 1).all()
