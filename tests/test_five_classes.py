import json

import numpy as np
import pandas as pd

import five_classes as F5
import six_classes as S


def test_merge_maps_cargo_and_tanker_to_one_class():
    assert [F5.CLASSES5[i] for i in F5.SIX_TO_FIVE] == ["background", "cargo + tanker", "fishing", "pleasurecraft",
                                                        "cargo + tanker", "tug"]
    p6 = np.array([[0.1, 0.3, 0.1, 0.1, 0.3, 0.1]])
    np.testing.assert_allclose(F5.merge_probs(p6), [[0.1, 0.6, 0.1, 0.1, 0.1]])


def test_fold5_keeps_six_class_weights_and_test_rows():
    table = pd.DataFrame([{"label": c, "MMSI": float(1000 * i + s), "date": 20220101 + s}
                          for i, c in enumerate(S.CLASSES6) for s in range(5 + 3 * i) for _ in range(1 + i % 3)])
    folds = S.make_folds(table)
    six, five = S.Fold(table, folds, 2), F5.Fold5(table, folds, 2)
    assert np.array_equal(six.rows["test"], five.rows["test"])
    np.testing.assert_allclose(six.w["train"], five.w["train"])
    assert set(np.unique(five.y["train+validation"])) <= set(range(5))
    assert np.array_equal(F5.SIX_TO_FIVE[five.y6["test"]], five.y["test"])


def test_run_all_on_synthetic_features(tmp_path):
    rng = np.random.default_rng(0)
    table = pd.DataFrame([{"label": c, "MMSI": float(1000 * i + s), "date": 20220101 + s}
                          for i, c in enumerate(S.CLASSES6) for s in range(10 + 4 * i) for _ in range(2)])
    table.to_parquet(tmp_path / "clips.parquet")
    rows = np.arange(len(table))
    y = table.label.map({c: i for i, c in enumerate(S.CLASSES6)}).to_numpy()
    eng_names = list(S.engineered_features(rng.normal(size=S.SR * 5).astype(np.float32)))
    for name, cols in [(S.SIMPLE_FILE, [f"m{i}" for i in range(20)]), (S.ENGINEERED_FILE, eng_names)]:
        X = rng.normal(size=(len(rows), len(cols)))
        X[:, :5] += 3 * np.eye(5)[F5.SIX_TO_FIVE[y]]                       # the five classes are separable
        pd.DataFrame(X, columns=cols).assign(row=rows).to_parquet(tmp_path / name)
    summary = F5.run_all(tmp_path, tmp_path / "out")
    res = pd.DataFrame(summary["results"])
    assert len(res) == S.N_FOLDS * 2 * 3                                     # 2 feature sets x (2 models + average)
    lr = res[res.model == "logistic regression"]
    assert lr.balanced_accuracy.mean() > 0.5 and set(lr.per_class_recall.iloc[0]) == set(F5.CLASSES5)
    preds = pd.read_parquet(tmp_path / "out" / "predictions.parquet")
    assert {"y6", "ship", "p_cargo + tanker"} <= set(preds.columns)
    assert json.loads((tmp_path / "out" / "results.json").read_text())["classes"] == F5.CLASSES5
