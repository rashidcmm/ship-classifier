import numpy as np
import pandas as pd

import merged_classes as M
import six_classes as S


def test_merged_weights_give_every_merged_class_the_same_total():
    labels = ["background"] * 4 + ["cargo"] * 6 + ["tanker"] * 2 + ["fishing"] * 3 + ["tug"] * 5 + ["pleasurecraft"] * 2
    y6 = np.array([S.CLASSES6.index(c) for c in labels])
    ships = [f"s{i}" for i in range(len(labels))]
    for grouping in M.GROUPINGS.values():
        w = M.merged_weights(y6, ships, grouping)
        y = M.mapping(grouping)[y6]
        totals = pd.Series(w).groupby(y).sum()
        np.testing.assert_allclose(totals, totals.iloc[0])               # every merged class the same
        if "cargo + tanker" in grouping:                                 # cargo and tanker half each
            np.testing.assert_allclose(w[y6 == 1].sum(), w[y6 == 4].sum())


def test_run_all_on_synthetic_features(tmp_path):
    rng = np.random.default_rng(0)
    table = pd.DataFrame([{"label": c, "MMSI": float(1000 * i + s), "date": 20220101 + s}
                          for i, c in enumerate(S.CLASSES6) for s in range(12) for _ in range(3)])
    table.to_parquet(tmp_path / "clips.parquet")
    rows = np.arange(len(table))
    y = table.label.map({c: i for i, c in enumerate(S.CLASSES6)}).to_numpy()
    X = rng.normal(size=(len(rows), 4)) + 2 * np.eye(6)[y][:, :4]
    pd.DataFrame(X, columns=["mfcc_1_mean", "third_100Hz", "shape_centroid_mean", "demon_depth"]).assign(
        row=rows).to_parquet(tmp_path / S.ENGINEERED_FILE)                # few columns: binning is slow here
    pd.DataFrame({"m0": rng.normal(size=len(rows)), "row": rows}).to_parquet(tmp_path / S.SIMPLE_FILE)
    s = M.run_all(tmp_path, tmp_path / "out", models=("SVM",))
    res = pd.DataFrame(s["results"])
    assert len(res) == 2 * S.N_FOLDS and res.balanced_accuracy.mean() > 0.4
    assert set(res[res.grouping == "4 classes"].per_class_recall.iloc[0]) == {"background", "cargo + tanker",
                                                                              "fishing + tug", "pleasurecraft"}
