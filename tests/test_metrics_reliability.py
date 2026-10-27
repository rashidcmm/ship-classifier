import numpy as np
import pytest
import torch

import metrics as M
import reliability as R
from models import ShipNet


def test_macro_f1_perfect_and_skip_absent():
    y = np.array([0, 0, 1, 1])
    assert M.macro_f1(y, y, n_classes=3) == pytest.approx(1.0)          # class 2 absent: skipped
    assert M.macro_f1(y, np.array([0, 1, 1, 1]), n_classes=3) == pytest.approx((2 / 3 + 0.8) / 2)
    assert M.accuracy(y, np.array([0, 1, 1, 1])) == pytest.approx(0.75)
    rec = M.per_class_recall(y, np.array([0, 1, 1, 1]), 3)
    assert rec[0] == pytest.approx(0.5) and rec[1] == 1.0 and np.isnan(rec[2])
    assert M.confusion(y, np.array([0, 1, 1, 1]), 3).tolist() == [[1, 1, 0], [0, 2, 0], [0, 0, 0]]


def test_ece_perfectly_calibrated_is_zero():
    probs = np.tile([0.8, 0.2], (100, 1))
    y = np.array([0] * 80 + [1] * 20)
    assert M.ece(probs, y, n_bins=15) < 0.01
    bins = M.reliability_bins(probs, y, n_bins=15)
    assert bins["count"].sum() == 100
    assert M.ece(np.tile([0.99, 0.01], (100, 1)), y) == pytest.approx(0.19, abs=1e-6)


def test_nll_and_brier():
    probs = np.array([[1.0, 0.0], [0.5, 0.5]])
    y = np.array([0, 1])
    assert M.nll(probs, y) == pytest.approx(-np.log(0.5) / 2, rel=1e-4)
    assert M.brier(probs, y) == pytest.approx((0 + 0.5) / 2)


def test_auroc_and_fpr95():
    score = np.r_[np.linspace(0, 0.4, 50), np.linspace(0.6, 1, 50)]
    unknown = np.r_[np.zeros(50, bool), np.ones(50, bool)]
    assert M.open_set_auroc(score, unknown) == pytest.approx(1.0)
    assert M.fpr_at_95_tpr(score, unknown) == pytest.approx(0.0)
    assert M.open_set_auroc(np.ones(100), unknown) == pytest.approx(0.5)
    assert M.fpr_at_95_tpr(np.ones(100), unknown) == pytest.approx(1.0)


def test_open_set_accuracy_counts():
    pred = np.array([0, 1, 2, 0])
    score = np.array([0.1, 0.9, 0.2, 0.8])
    y = np.array([0, -1, 1, -1])
    assert M.open_set_accuracy(pred, score, 0.5, y) == pytest.approx(0.75)


def test_regression_report_bands():
    rep = M.regression_report(np.array([100.0, 800, 1500]), np.array([150.0, 1450, 1600]), bands=[700, 1400])
    assert rep["mae"] == pytest.approx((50 + 650 + 100) / 3)
    assert rep["median_ae"] == pytest.approx(100)
    assert rep["band_accuracy"] == pytest.approx(2 / 3)
    assert rep["rmse"] == pytest.approx(np.sqrt((50 ** 2 + 650 ** 2 + 100 ** 2) / 3))



def test_distance_bin_report():
    # 250 m bins up to 2 km: true bins 0, 2, 5, 7; predicted bins 0, 3, 5, 4
    rep = M.distance_bin_report(np.array([100.0, 600, 1300, 1900]), np.array([200.0, 900, 1300, 1000]),
                                width=250, top=2000, near=700)
    assert rep["n_bins"] == 8
    assert rep["bin_accuracy"] == pytest.approx(2 / 4)
    assert rep["within_one_bin"] == pytest.approx(3 / 4)
    # near (< 700 m): 1 of 2 found; far: 2 of 2 found
    assert rep["near_far_balanced_accuracy"] == pytest.approx((1 / 2 + 2 / 2) / 2)


def test_distance_bin_report_last_bin_is_open_ended():
    rep = M.distance_bin_report(np.array([1900.0, 2000.0]), np.array([5000.0, 1800.0]), width=250, top=2000, near=700)
    assert rep["bin_accuracy"] == pytest.approx(1.0)
    assert rep["near_far_balanced_accuracy"] == pytest.approx(1.0)       # only far clips: averaged over far only

def test_energy_lower_for_confident():
    logits = np.array([[10.0, 0, 0], [1, 1, 1]])
    e = R.energy_score(logits)
    assert e[0] < e[1]
    msp = R.msp_score(logits)
    assert msp[0] < msp[1] and msp[1] == pytest.approx(2 / 3)
    assert np.allclose(R.softmax(logits).sum(1), 1)


def test_entropy_and_mutual_information():
    uniform = np.full((1, 4), 0.25)
    assert R.entropy(uniform)[0] == pytest.approx(np.log(4))
    same = np.repeat(np.array([[[0.7, 0.3]]]), 5, axis=0)            # (S=5, N=1, C=2), samples agree
    assert R.mutual_information(same)[0] == pytest.approx(0.0, abs=1e-9)
    disagree = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    assert R.mutual_information(disagree)[0] == pytest.approx(np.log(2), rel=1e-3)
    assert np.allclose(R.ensemble_probs([np.array([[1.0, 0]]), np.array([[0.0, 1]])]), [[0.5, 0.5]])


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, size=(6000, 5))
    p = R.softmax(z)
    y = np.array([rng.choice(5, p=row) for row in p])           # labels drawn from the logits' own softmax: calibrated
    assert R.fit_temperature(z * 3, y) == pytest.approx(3.0, abs=0.3)
    assert R.fit_temperature(z, y) == pytest.approx(1.0, abs=0.15)


def test_mc_dropout_varies():
    torch.manual_seed(0)
    net = ShipNet("tiny_cnn", n_classes=4, pretrained=False, drop_rate=0.5).eval()
    bn = next(m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d))
    before = bn.running_mean.clone()
    loader = [{"x": torch.randn(6, 1, 128, 32)}, {"x": torch.randn(3, 1, 128, 32)}]
    s = R.mc_dropout_logits(net, loader, n_samples=5, device=torch.device("cpu"))
    assert s.shape == (5, 9, 4)
    assert not np.allclose(s[0], s[1])
    assert torch.equal(bn.running_mean, before)
    assert not net.training and not net.dropout.training


def test_fit_temperature_weights():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, size=(4000, 4))
    y = np.array([rng.choice(4, p=row) for row in R.softmax(z)])
    logits = np.r_[z * 3, z]                              # first half overconfident, second half calibrated
    yy = np.r_[y, y]
    ones = np.ones(len(yy))
    assert R.fit_temperature(logits, yy, weights=ones) == pytest.approx(R.fit_temperature(logits, yy), rel=1e-6)
    only_first = np.r_[np.ones(4000), np.zeros(4000)]
    assert R.fit_temperature(logits, yy, weights=only_first) == pytest.approx(R.fit_temperature(z * 3, y), rel=1e-4)


def test_weighted_percentile():
    v = np.arange(1, 101, dtype=float)
    assert M.weighted_percentile(v, np.ones(100), 95) == pytest.approx(np.percentile(v, 95), abs=1.0)
    w = np.r_[np.ones(50), np.zeros(50)]                  # values 51..100 ignored
    assert M.weighted_percentile(v, w, 100) == pytest.approx(50)
    assert M.weighted_percentile(np.array([1.0, 2, 3]), np.array([1.0, 1, 2]), 50) == pytest.approx(2.5, abs=0.5)


def test_class_balanced_weights():
    y = np.array([0] * 90 + [1] * 10)
    w = M.class_balanced_weights(y)
    assert w[y == 0].sum() == pytest.approx(w[y == 1].sum())
    assert w.sum() == pytest.approx(len(y))
