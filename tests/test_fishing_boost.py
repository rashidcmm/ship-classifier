import numpy as np
import pandas as pd

import fishing_boost as FB
import six_classes as S


def test_offsets_raise_an_under_called_class():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 100)
    probs = np.column_stack([np.full(200, 0.6), np.full(200, 0.4)])
    probs[100:, 1] += rng.uniform(0, 0.15, 100)                              # class 1 is a bit more likely ...
    probs[100:, 0] = 1 - probs[100:, 1]                                      # ... but never wins
    off = FB.best_offsets(probs, y)
    assert off[0] == 0 and off[1] > 0
    assert (FB.apply_offsets(probs, off).argmax(1)[100:] == 1).mean() > 0.5
    np.testing.assert_allclose(FB.apply_offsets(probs, np.zeros(2)), probs, atol=1e-6)


def test_run_all_on_synthetic_features(tmp_path):
    rng = np.random.default_rng(0)
    table = pd.DataFrame([{"label": c, "MMSI": float(1000 * i + s), "date": 20220101 + s}
                          for i, c in enumerate(S.CLASSES6) for s in range(30) for _ in range(3)])
    table.to_parquet(tmp_path / "clips.parquet")
    rows = np.arange(len(table))
    y = table.label.map({c: i for i, c in enumerate(S.CLASSES6)}).to_numpy()
    X = rng.normal(size=(len(rows), 4)) + np.eye(6)[y][:, :4] * 2
    pd.DataFrame(X, columns=["mfcc_1_mean", "third_100Hz", "shape_centroid_mean", "demon_depth"]).assign(
        row=rows).to_parquet(tmp_path / S.ENGINEERED_FILE)                   # few columns: binning is slow here
    pd.DataFrame({"m0": rng.normal(size=len(rows)), "row": rows}).to_parquet(tmp_path / S.SIMPLE_FILE)
    s = FB.run_all(tmp_path, tmp_path / "out", leaves=(7,), min_leaf=(10,))
    res = pd.DataFrame(s["results"])
    assert set(res.version) == {"lesson 16 setting", "tuned trees", "tuned trees + class offsets"}
    assert len(res) == 3 * S.N_FOLDS and res.balanced_accuracy.mean() > 0.4
