import json

import numpy as np
import pandas as pd

import six_classes as S
import six_classes_fewer as B
from baseline_classical import band_stats


def test_bands_below_and_merging_counts():
    assert B.bands_below(16000) == 128
    counts = [B.bands_below(t) for t in B.TOP_HZ]
    assert counts == sorted(counts, reverse=True) and counts[-1] > 20        # 1 kHz still keeps a few dozen bands
    assert B.n_groups(16000, 8) == 16 and B.n_groups(1000, 1) == B.bands_below(1000)
    lo, hi = B.group_edges(1)[B.bands_below(1000) - 1]
    assert hi <= 1000 + 1e-6 < B.group_edges(1)[B.bands_below(1000)][1]      # last kept band ends below 1 kHz


def test_band_features_match_lesson13_simple_set_and_count_mfccs():
    rng = np.random.default_rng(0)
    mel = rng.normal(-60, 10, size=(7, 128, 30)).astype(np.float16)
    df = B.band_features(mel, np.arange(7), chunk=3)
    simple = band_stats(mel)                                                 # lesson 13: 128 means, 128 stds, 1 mean
    cols = B.variant_columns(df.columns, 16000, 1, "bands")
    assert len(cols) == 256
    np.testing.assert_allclose(df[[f"band1_{g}_mean" for g in range(128)]].to_numpy(), simple[:, :128], rtol=1e-5)
    np.testing.assert_allclose(df[[f"band1_{g}_std" for g in range(128)]].to_numpy(), simple[:, 128:256],
                               rtol=1e-4, atol=1e-4)
    assert len(B.variant_columns(df.columns, 16000, 1, "mfcc")) == 3 * 19
    k = min(19, B.n_groups(1000, 8) - 1)
    assert len(B.variant_columns(df.columns, 1000, 8, "mfcc")) == 3 * k
    # a merged band of a flat spectrum is the flat level plus 10*log10(number of bands merged)
    flat = np.full((1, 128, 10), -50.0, dtype=np.float16)
    m4 = B.band_features(flat, [0])[[f"band4_{g}_mean" for g in range(32)]].to_numpy()
    np.testing.assert_allclose(m4, -50 + 10 * np.log10(4), atol=1e-3)


def _synthetic(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    table = pd.DataFrame([{"label": c, "MMSI": float(1000 * i + s), "date": 20220101 + s}
                          for i, c in enumerate(S.CLASSES6) for s in range(10 + 4 * i) for _ in range(2)])
    table.to_parquet(tmp_path / "clips.parquet")
    y = table.label.map({c: i for i, c in enumerate(S.CLASSES6)}).to_numpy()
    return rng, table, y


def test_run_bands_on_synthetic_mel(tmp_path):
    rng, table, y = _synthetic(tmp_path)
    mel = rng.normal(-60, 3, size=(len(table), 128, 12)).astype(np.float32)
    mel[:, :20] += 6 * y[:, None, None]                                      # the class shows in the low bands
    B.band_features(mel, np.arange(len(table))).to_parquet(tmp_path / B.BANDS_FILE)
    summary = B.run_bands(tmp_path, tmp_path / "out", tops=(16000, 1000), merges=(1, 4),
                          models=("logistic regression",))
    res = pd.DataFrame(summary["results"])
    assert len(res) == 2 * 2 * 2 * S.N_FOLDS                                 # 4 versions x 2 kinds x 5 folds
    ba = res.groupby(["kind", "top_hz", "merge"]).balanced_accuracy.mean()
    assert ba.min() > 0.3                                                    # chance is 1/6
    assert ba["bands", 1000, 4] > ba["bands", 16000, 1]                      # dropping noise-only high bands helps
    # a second call continues the saved run instead of starting again
    again = B.run_bands(tmp_path, tmp_path / "out", tops=(16000, 1000), merges=(1, 4),
                        models=("logistic regression",))
    assert len(again["results"]) == len(res)


def test_run_groups_on_synthetic_features(tmp_path):
    rng, table, y = _synthetic(tmp_path, seed=1)
    rows = np.arange(len(table))
    names = list(S.engineered_features(rng.normal(size=S.SR * 5).astype(np.float32)))
    X = rng.normal(size=(len(rows), len(names)))
    X[:, names.index("third_100Hz")] += 3 * y                                # only the 1/3-octave group helps
    pd.DataFrame(X, columns=names).assign(row=rows).to_parquet(tmp_path / S.ENGINEERED_FILE)
    pd.DataFrame(rng.normal(size=(len(rows), 3)), columns=["a", "b", "c"]).assign(row=rows).to_parquet(
        tmp_path / S.SIMPLE_FILE)
    summary = B.run_groups(tmp_path, tmp_path / "out", models=("logistic regression",))
    res = pd.DataFrame(summary["results"])
    logo = res[res.kind == "leave one out"]
    assert logo.left_out.nunique() == 8                                      # none + each of the 7 groups
    fwd = res[res.kind == "forward"]
    assert sorted(fwd.step.unique()) == list(range(1, 8))
    assert fwd[fwd.step == 1].added.iloc[0] == "1/3-octave levels"           # the useful group is found first
    forward = pd.read_csv(tmp_path / "out" / "groups_forward.csv")
    assert forward.groupby("step").chosen.sum().eq(1).all()
    assert json.loads((tmp_path / "out" / "groups_results.json").read_text())["groups"]["MFCC"] == 57
