import json

import numpy as np
import pandas as pd

import open_set as O
import panns_features as P
import six_classes as S
import with_other as W


def test_mel_statistics_columns_and_values():
    rng = np.random.default_rng(0)
    x = rng.normal(-60, 10, size=(3, 128, 50)).astype(np.float16)
    st = W.mel_statistics(x)
    assert st.shape == (3, 2 * 128 + 1 + 3 * 128 + 2 * (64 + 32 + 16))
    np.testing.assert_allclose(st["mel5_p50"], np.median(x[:, 5].astype(np.float32), axis=1), rtol=1e-5)
    # two equal bands merged: +3 dB, same spread
    y = np.repeat(x[:, :1], 128, axis=1)
    st = W.mel_statistics(y)
    np.testing.assert_allclose(st["band2_0_mean"], st["mel0_mean"] + 10 * np.log10(2), atol=1e-3)


def test_band_variants_and_lowest_bands_top_frequency():
    v = W.band_variants()
    assert len(v["128 bands, 10 Hz-16 kHz"]) == 256 and len(v[[k for k in v if k.startswith("lowest 20 ")][0]]) == 40
    tops = [W.top_hz_of_lowest(n) for n in W.LOWEST]
    assert tops == sorted(tops) and 300 < tops[0] < 700


def test_subset_cost_counts_shared_steps_once():
    costs = {"read and decode": 1.0, "Welch spectrum": 10.0, "STFT (2,048 window)": 20.0,
             "1/3-octave levels": 2.0, "tonal lines": 3.0, "propeller modulation (DEMON)": 5.0}
    assert W.subset_cost(["1/3-octave levels", "tonal lines"], costs) == 1 + 10 + 2 + 3
    assert W.subset_cost(["propeller modulation (DEMON)"], costs) == 1 + 5


def _synthetic(tmp_path, rng):
    labels = S.CLASSES6 + O.LEFT_OUT
    table = pd.DataFrame([{"label": c, "MMSI": float(1000 * i + s), "date": 20220101 + s, "zip_name": "x"}
                          for i, c in enumerate(labels) for s in range(10) for _ in range(3)])
    table.to_parquet(tmp_path / "clips.parquet")
    y = np.array([labels.index(c) for c in table.label])
    rows = np.arange(len(table))
    six = table.label.isin(S.CLASSES6).to_numpy()
    eng_cols = ["mfcc_1_mean", "third_100Hz", "shape_centroid_mean", "demon_depth"]
    X = rng.normal(size=(len(table), 4)) + 2 * np.eye(12)[y][:, [1, 2, 3, 5]]
    pd.DataFrame(X[six], columns=eng_cols).assign(row=rows[six]).to_parquet(tmp_path / S.ENGINEERED_FILE)
    pd.DataFrame(X[~six], columns=eng_cols).assign(row=rows[~six]).to_parquet(tmp_path / P.LEFTOUT_FILE)
    pd.DataFrame({"m0": rng.normal(size=six.sum()), "row": rows[six]}).to_parquet(tmp_path / S.SIMPLE_FILE)
    mel =rng.normal(-60, 5, size=(len(table), 128, 20)).astype(np.float32)
    mel[np.arange(len(table)), y * 10] += 15                                  # every label is loud in its own band
    W.mel_statistics(mel).assign(row=rows).to_parquet(tmp_path / W.MEL_FILE)
    emb = rng.normal(size=(len(table), 16))
    emb[:, :12] += np.eye(12)[y]
    np.save(tmp_path / P.EMB_FILE, emb.astype(np.float16))
    np.save(tmp_path / P.EMB_ROWS_FILE, rows)
    return table


def test_all_steps_on_synthetic_data(tmp_path, monkeypatch):
    # every gradient-boosting fit has a fixed start-up cost of seconds on the laptop: keep the steps small
    full_sets = W.feature_sets
    monkeypatch.setattr(W, "feature_sets", lambda eng_cols: dict(list(full_sets(eng_cols).items())[1:3]))
    monkeypatch.setattr(W, "band_variants", lambda: {"16 bands": [f"band8_{j}_mean" for j in range(16)]})
    monkeypatch.setattr(W, "GROUPINGS", {k: W.GROUPINGS[k] for k in ("6 classes + other", "4 classes + other")})
    monkeypatch.setattr(W, "MODELS", ("logistic regression",))       # gradient boosting: see the next test
    monkeypatch.setattr(W, "N_PCA", 4)
    monkeypatch.setattr(W, "cost_groups", lambda cols: {"MFCC": ["mfcc_1_mean"], "tonal lines": ["third_100Hz"],
                                                        "spectral shape": ["shape_centroid_mean"]})
    monkeypatch.setattr(W, "measure_costs", lambda table, rows: {
        "read and decode": 1.0, "mel spectrogram (128 bands)": 8.0, "STFT (2,048 window)": 4.0,
        "Welch spectrum": 3.0, "MFCC": 1.0, "tonal lines": 1.0, "spectral shape": 0.5})
    _synthetic(tmp_path, np.random.default_rng(0))
    out = tmp_path / "out"
    W.run(W.STEPS, tmp_path, out)

    main = pd.DataFrame(W.Log(out / "main.jsonl").rows)
    assert len(main) == 2 * S.N_FOLDS and set(main.grouping) == {W.MAIN}
    assert "other" in main.per_class_recall.iloc[0] and len(main.per_class_recall.iloc[0]) == 7
    preds = pd.read_parquet(out / "main_predictions.parquet")
    assert preds.groupby(["features", "model"]).row.nunique().eq(len(pd.read_parquet(tmp_path / "clips.parquet"))).all()

    assert len(W.Log(out / "bands.jsonl").rows) == S.N_FOLDS
    walk = pd.DataFrame(W.Log(out / "walk.jsonl").rows)
    assert (walk.stage == "alone").sum() == 3 and (walk.stage == "walk").sum() == 2     # 3 sets alone, 2 additions
    result = json.loads((out / "walk_result.json").read_text())
    assert result["final"].split(W.SEP)[0] == result["order"][0]
    added = walk[walk.stage == "walk"]
    assert all((row.val > row.before) == (row.added in result["final"].split(W.SEP)) for row in added.itertuples())

    merged = pd.DataFrame(W.Log(out / "merged.jsonl").rows)
    assert len(merged) == 2 * S.N_FOLDS + 2 * S.N_FOLDS                   # 2 groupings, then PANN alone and + chosen set
    assert set(merged.grouping) == set(W.GROUPINGS)
    assert {"PANN (256)"} <= set(merged.features)
    n_rows_before = len(merged)
    W.run(("merged",), tmp_path, out)                                      # resumes: nothing is run twice
    assert len(W.Log(out / "merged.jsonl").rows) == n_rows_before


def test_gradient_boosting_run_is_timed(tmp_path):
    _synthetic(tmp_path, np.random.default_rng(1))
    table, folds, feats, eng_cols = W.load_all(tmp_path)
    r = W.Runner(table, folds, feats, tmp_path / "out")
    log = W.Log(tmp_path / "out" / "x.jsonl")
    r.run("x", log, W.MAIN, "engineered", list(eng_cols), "gradient boosting", 1)
    rec = log.rows[0]
    assert rec["setting"] >= 1 and rec["seconds_fit"] > 0 and rec["predict_us_per_clip"] > 0
    assert set(rec["per_class_recall"]) == set(W.GROUPINGS[W.MAIN])


def test_two_processes_share_a_log(tmp_path):
    a, b = W.Log(tmp_path / "x.jsonl"), W.Log(tmp_path / "x.jsonl")
    assert a.add({"fold": 1, "v": 1}, {"fold": 1})
    assert b.has(fold=1) and not b.add({"fold": 1, "v": 2}, {"fold": 1})      # the second process sees the first
    assert len(W.Log(tmp_path / "x.jsonl").rows) == 1
