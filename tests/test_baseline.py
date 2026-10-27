import json

import numpy as np
import pytest
from scipy.fft import dct

import baseline_classical as B


def test_band_stats_shape():
    block = np.random.default_rng(0).normal(size=(5, 128, 64)).astype(np.float16)
    f = B.band_stats(block)
    assert f.shape == (5, 257)
    assert f[0, :128] == pytest.approx(block[0].astype(np.float64).mean(axis=1), rel=1e-4)
    assert f[0, 256] == pytest.approx(block[0].astype(np.float64).mean(), rel=1e-4)


def test_baseline_runs_on_mini(mini_data, tmp_path):
    res = B.run_baseline(mini_data, fold=1, out_dir=tmp_path / "baseline_f1")
    saved = json.loads((tmp_path / "baseline_f1" / "metrics.json").read_text())
    assert saved["class"]["accuracy"] > 1 / 12
    assert saved["distance"]["mae"] > 1 and saved["speed"]["mae"] > 0      # metres / knots
    assert res["class"]["macro_f1"] == saved["class"]["macro_f1"]


def test_rich_stats_adds_mfcc_and_percentiles():
    block = np.random.default_rng(0).normal(size=(3, 128, 64)).astype(np.float16)
    f = B.rich_stats(block)
    assert f.shape == (3, 257 + 40 + 3 * 128 + 3 * 20 + 20)
    assert f[:, :257] == pytest.approx(B.band_stats(block), rel=1e-4, abs=1e-4)
    x = block[0].astype(np.float64)
    mfcc = dct(x, type=2, norm="ortho", axis=0)[:20]                    # MFCC = DCT of the log-mel bands
    assert f[0, 257:277] == pytest.approx(mfcc.mean(axis=1), rel=1e-3, abs=1e-3)
    assert f[0, 297:425] == pytest.approx(np.percentile(x, 10, axis=1), rel=1e-3, abs=1e-3)
    assert f[0, -20:] == pytest.approx(np.diff(mfcc, axis=1).std(axis=1), rel=1e-3, abs=1e-3)


def test_baseline_rich_features_and_distance_bins(mini_data, tmp_path):
    B.main(["--data-dir", str(mini_data), "--runs-dir", str(tmp_path), "--features", "rich"])
    saved = json.loads((tmp_path / "baseline_rich_f1" / "metrics.json").read_text())
    assert "761" in saved["features"]
    assert "within_one_bin" in saved["distance"]


def test_baseline_merge_into_other(mini_data, tmp_path):
    B.main(["--data-dir", str(mini_data), "--runs-dir", str(tmp_path), "--merge-into-other", "pilotvessel", "dredger", "rescue"])
    saved = json.loads((tmp_path / "baseline_merged_f1" / "metrics.json").read_text())
    recall = saved["class"]["per_class_recall"]
    assert len(recall) == 9 and "other" in recall and "pilotvessel" not in recall
    assert saved["merge_into_other"] == ["pilotvessel", "dredger", "rescue"]
