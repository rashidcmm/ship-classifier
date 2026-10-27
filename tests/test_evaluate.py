import json

import numpy as np
import pandas as pd
import pytest

import data
import evaluate
import train
from conftest import make_mini_data


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("eval")
    mini = make_mini_data(root / "data")

    def tr(*extra):
        argv = ["--data-dir", str(mini), "--runs-dir", str(root / "runs"), "--backbone", "tiny_cnn",
                "--no-pretrained", "--workers", "0", "--device", "cpu", "--lr", "3e-3", *extra]
        return train.run(train.make_parser().parse_args(argv))

    return {"data": mini,
            "cls": tr("--task", "class", "--epochs", "6"),
            "cls_s1": tr("--task", "class", "--epochs", "6", "--seed", "1"),
            "hold": tr("--task", "class", "--epochs", "6", "--holdout", "A"),
            "dist": tr("--task", "distance", "--epochs", "2")}


def test_evaluate_closed_set_outputs(runs):
    res = evaluate.evaluate_run(runs["cls"], device="cpu")
    m = json.loads((runs["cls"] / "eval" / "metrics.json").read_text())
    for key in ("accuracy", "macro_f1", "per_class_recall", "n_test", "rare_class_warning"):
        assert key in m["closed_set"]
    assert set(m["calibration"]) >= {"ece_before", "ece_after", "nll_before", "nll_after", "brier_before", "brier_after"}
    assert "by_distance" in m and res["closed_set"]["n_test"] == 12
    assert (runs["cls"] / "eval" / "confusion.png").exists()
    assert (runs["cls"] / "eval" / "reliability_before_after.png").exists()
    cal = json.loads((runs["cls"] / "eval" / "calibration.json").read_text())
    assert cal["temperature"] > 0 and {"msp_threshold", "energy_threshold"} <= set(cal)


def test_thresholds_from_validation_not_test(runs, monkeypatch):
    evaluate.evaluate_run(runs["cls"], device="cpu")
    before = (runs["cls"] / "eval" / "calibration.json").read_text()
    original = data.open_features

    def shuffled_test_labels(data_dir):
        mel, table = original(data_dir)
        test = np.flatnonzero(table.split_f1.to_numpy() == "test")
        table = table.copy()
        table.loc[test, "label"] = np.random.default_rng(1).permutation(table.label.to_numpy()[test])
        return mel, table

    monkeypatch.setattr(evaluate.D, "open_features", shuffled_test_labels)
    evaluate.evaluate_run(runs["cls"], device="cpu")
    assert (runs["cls"] / "eval" / "calibration.json").read_text() == before


def test_evaluate_holdout_open_set(runs):
    res = evaluate.evaluate_run(runs["hold"], device="cpu")
    os_ = res["open_set"]
    for score in ("msp", "energy"):
        assert 0.0 <= os_[score]["auroc"] <= 1.0
        assert 0.0 <= os_[score]["fpr95"] <= 1.0
    assert "open_set_accuracy_energy" in os_
    assert (runs["hold"] / "eval" / "open_set_roc.png").exists()


def test_evaluate_ensemble_and_mc(runs):
    res = evaluate.evaluate_run(runs["cls"], ensemble=[runs["cls_s1"]], mc_samples=3, device="cpu")
    pred = pd.read_parquet(runs["cls"] / "eval" / "predictions.parquet")
    assert {"ens_entropy", "mc_entropy", "mc_mi"} <= set(pred.columns)
    assert "ensemble" in res and "mc_dropout" in res


def test_evaluate_regression(runs):
    res = evaluate.evaluate_run(runs["dist"], device="cpu")
    reg = res["regression"]["distance"]
    assert reg["mae"] > 1 and reg["baseline_mae"] > 1          # metres, not km
    assert "band_accuracy" in reg
    assert {"bin_accuracy", "within_one_bin", "near_far_balanced_accuracy"} <= set(reg)
    assert "within_one_bin" in reg["baseline_bins"]                    # the always-median guess, scored the same way
    assert (runs["dist"] / "eval" / "distance_scatter.png").exists()


def test_add_noise_snr():
    rng = np.random.default_rng(0)
    wave = np.sin(np.linspace(0, 2000, 160_000)).astype(np.float32)
    noisy = evaluate.add_noise(wave, 10.0, rng)
    snr = 10 * np.log10(np.mean(wave ** 2) / np.mean((noisy - wave) ** 2))
    assert snr == pytest.approx(10.0, abs=0.2)


def test_calibrate_invariant_to_class_duplication():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 2, size=(80, 4))
    y = np.repeat(np.arange(4), 20)
    base = evaluate.calibrate(logits, y)
    dup = np.flatnonzero(y == 0)
    flooded = evaluate.calibrate(np.r_[logits, np.repeat(logits[dup], 10, axis=0)], np.r_[y, np.repeat(y[dup], 10)])
    assert flooded["temperature"] == pytest.approx(base["temperature"], rel=1e-3)
    for k in ("msp_threshold", "energy_threshold"):
        assert flooded[k] == pytest.approx(base[k], abs=1e-9)
    assert "class-balanced" in base["chosen_on"]


def test_evaluate_keeps_earlier_sections(runs):
    evaluate.evaluate_run(runs["cls"], ensemble=[runs["cls_s1"]], mc_samples=2, device="cpu")
    res = evaluate.evaluate_run(runs["cls"], device="cpu")                # e.g. a later --noise call
    assert "mc_dropout" in res and "ensemble" in res
    pred = pd.read_parquet(runs["cls"] / "eval" / "predictions.parquet")
    assert {"mc_mi", "ens_entropy"} <= set(pred.columns)


def test_evaluate_drops_sections_from_old_model(runs):
    evaluate.evaluate_run(runs["cls"], mc_samples=2, device="cpu")
    best = runs["cls"] / "best.pt"
    best.write_bytes(best.read_bytes())                                   # "retrained": new file timestamp
    import os, time
    os.utime(best, ns=(time.time_ns(), time.time_ns() + 10**9))
    res = evaluate.evaluate_run(runs["cls"], device="cpu")
    assert "mc_dropout" not in res


def test_noise_without_zip_fails_before_work(runs, monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate, "predict_rows", lambda *a, **k: pytest.fail("evaluation ran before the zip check"))
    with pytest.raises(FileNotFoundError, match="audio zip"):
        evaluate.evaluate_run(runs["cls"], noise_snrs=(10,), audio_zip=tmp_path / "missing.zip", device="cpu")


def test_ensemble_refuses_other_fold(runs):
    import train as T
    argv = ["--data-dir", str(runs["data"]), "--runs-dir", str(runs["cls"].parent), "--backbone", "tiny_cnn",
            "--no-pretrained", "--workers", "0", "--device", "cpu", "--task", "class", "--epochs", "1", "--fold", "2"]
    other = T.run(T.make_parser().parse_args(argv))
    with pytest.raises(ValueError, match="fold"):
        evaluate.evaluate_run(runs["cls"], ensemble=[other], device="cpu")
