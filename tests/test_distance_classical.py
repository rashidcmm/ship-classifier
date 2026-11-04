import json

import numpy as np
import pandas as pd

import distance_classical as DC
import six_classes as S


def test_buckets_and_scores():
    assert list(DC.bucket_of([0, 249, 250, 1999, 2000, 2500])) == [0, 0, 1, 7, 7, 7]
    s = DC.bucket_scores([0, 1, 2, 3], [0, 2, 2, 5])
    assert s["exact"] == 0.5 and s["near"] == 0.75                           # 1 -> 2 is "near", 3 -> 5 is not
    assert s["exact_500m"] == 0.5                                            # 500 m buckets: 0|0, 0|1, 1|1, 1|2


def test_feature_sets_differ_only_in_mfcc():
    names = list(S.engineered_features(np.random.default_rng(0).normal(size=S.SR * 5).astype(np.float32)))
    sets = DC.feature_sets(names)
    full, reduced = sets.values()
    assert len(full) == 144 and len(full) - len(reduced) == 3 * (19 - DC.REDUCED_MFCC)


def test_run_all_on_synthetic_features(tmp_path, monkeypatch):
    # gradient boosting's binning step costs ~0.1 s per column on this laptop: keep the synthetic sets small
    monkeypatch.setattr(DC, "feature_sets", lambda cols: {"small": ["level_broadband", "mfcc_1_mean", "mfcc_2_mean"],
                                                         "smaller": ["level_broadband"]})
    rng = np.random.default_rng(0)
    table = pd.DataFrame([{"label": c, "MMSI": float(1000 * i + s), "date": 20220101 + s}
                          for i, c in enumerate(S.CLASSES6) for s in range(30) for _ in range(3)])
    table["mean_distance"] = rng.uniform(0, 2000, len(table))
    table.to_parquet(tmp_path / "clips.parquet")
    rows = np.arange(len(table))
    names = list(S.engineered_features(rng.normal(size=S.SR * 5).astype(np.float32)))
    X = rng.normal(size=(len(rows), len(names)))
    X[:, names.index("level_broadband")] = -table.mean_distance / 100        # louder when closer
    pd.DataFrame(X, columns=names).assign(row=rows).to_parquet(tmp_path / S.ENGINEERED_FILE)
    pd.DataFrame({"m0": rng.normal(size=len(rows)), "row": rows}).to_parquet(tmp_path / S.SIMPLE_FILE)
    summary = DC.run_all(tmp_path, tmp_path / "out")
    assert len(summary["results"]) == S.N_FOLDS
    assert np.mean([r["regression"]["mae_m"] for r in summary["results"]]) < 300
    preds = pd.read_parquet(tmp_path / "out" / "predictions.parquet")
    assert (preds.label != "background").all() and set(json.loads((tmp_path / "out" / "results.json").read_text()))
