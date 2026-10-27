import json

import numpy as np
import pytest

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
