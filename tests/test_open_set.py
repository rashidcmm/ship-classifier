import numpy as np
import pandas as pd

import open_set as O
import panns_features as P
import six_classes as S


def _synthetic(tmp_path, rng):
    labels = S.CLASSES6 + O.LEFT_OUT
    table = pd.DataFrame([{"label": c, "MMSI": float(1000 * i + s), "date": 20220101 + s}
                          for i, c in enumerate(labels) for s in range(10) for _ in range(3)])
    table.to_parquet(tmp_path / "clips.parquet")
    y = np.array([labels.index(c) for c in table.label])
    cols = ["mfcc_1_mean", "third_100Hz", "shape_centroid_mean", "demon_depth"]
    X = rng.normal(size=(len(table), 4)) + 2 * np.eye(12)[y][:, [1, 2, 3, 5]]
    six = table.label.isin(S.CLASSES6).to_numpy()
    rows = np.arange(len(table))
    pd.DataFrame(X[six], columns=cols).assign(row=rows[six]).to_parquet(tmp_path / S.ENGINEERED_FILE)
    pd.DataFrame(X[~six], columns=cols).assign(row=rows[~six]).to_parquet(tmp_path / P.LEFTOUT_FILE)
    pd.DataFrame({"m0": rng.normal(size=six.sum()), "row": rows[six]}).to_parquet(tmp_path / S.SIMPLE_FILE)
    emb = rng.normal(size=(len(table), 16))
    emb[:, :12] += np.eye(12)[y]                                             # the class shows in the embedding
    np.save(tmp_path / P.EMB_FILE, emb.astype(np.float16))
    np.save(tmp_path / P.EMB_ROWS_FILE, rows)
    return table


def test_folds_keep_the_six_class_folds_and_split_left_out_ships():
    rng = np.random.default_rng(0)
    labels = S.CLASSES6 + O.LEFT_OUT
    table = pd.DataFrame([{"label": c, "MMSI": float(1000 * i + s), "date": 20220101 + s}
                          for i, c in enumerate(labels) for s in range(10) for _ in range(2)])
    folds = O.all_folds(table)
    np.testing.assert_array_equal(folds[table.label.isin(S.CLASSES6)], S.make_folds(table)[table.label.isin(S.CLASSES6)])
    left = table.assign(fold=folds, ship=S.ship_key(table))[table.label.isin(O.LEFT_OUT)]
    assert (left.fold > 0).all() and (left.groupby("ship").fold.nunique() == 1).all()


def test_run_all_on_synthetic_data(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "N_PCA", 4)
    _synthetic(tmp_path, np.random.default_rng(0))
    s = O.run_all(tmp_path, tmp_path / "out")
    res = pd.DataFrame(s["results"])
    assert set(res.experiment) == {"A", "B", "C"}
    b = res[res.experiment == "B"]
    assert len(b) == S.N_FOLDS and b.known_accepted.between(0, 1).all()
    a = res[res.experiment == "A"]
    assert "other" in a.per_class_recall.iloc[0]
    assert set(res[res.experiment == "C"].features) == {"PANN (256)", "PANN (256) + engineered (144)",
                                                        "engineered (144)"}
