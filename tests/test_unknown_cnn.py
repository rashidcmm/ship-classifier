import json

import numpy as np
import pandas as pd
import pytest
import torch

import fusion_cnn as FN
import open_set as O
import six_classes as S
import unknown_cnn as U
from models import save_checkpoint
from test_with_other import _synthetic

NAMES = ["a", "b", "c"]


def _onehot(classes, n=3, conf=0.9):
    p = np.full((len(classes), n), (1 - conf) / (n - 1))
    p[np.arange(len(classes)), classes] = conf
    return p


def test_threshold_keeps_the_chosen_share_of_known_scores():
    conf = np.linspace(0, 1, 101)
    t = U.threshold(conf, keep=0.95)
    assert (conf >= t).mean() == pytest.approx(0.95, abs=0.011)


def test_open_set_scores_when_unknown_ships_are_unsure():
    # known: 2 ships per class, each 2 clips, confident and right; unknown: 3 ships with flat probabilities
    ky = np.repeat([0, 1, 2], 4)
    known_ships = np.repeat([f"k{i}" for i in range(6)], 2)
    unknown_ships = np.repeat(["u0", "u1", "u2"], 2)
    out = U.open_set_scores(
        val_probs=_onehot(ky), val_ships=known_ships,
        known_probs=_onehot(ky), known_y=ky, known_ships=known_ships,
        unknown_probs=np.full((6, 3), 1 / 3), unknown_ships=unknown_ships,
        unknown_types=np.repeat(["sailing", "sailing", "dredger"], 2), names=NAMES, keep=1.0)
    assert out["auroc_ship"] == 1.0 and out["auroc_clip"] == 1.0
    assert out["known_ships_accepted"] == 1.0 and out["unknown_ships_rejected"] == 1.0
    assert out["unknown_ships_rejected_by_type"] == {"dredger": 1.0, "sailing": 1.0}
    assert out["open_set_per_ship_recall"] == {"a": 1.0, "b": 1.0, "c": 1.0, "unknown": 1.0}
    assert out["open_set_per_ship_balanced_accuracy"] == 1.0
    assert out["closed_per_ship_balanced_accuracy"] == 1.0
    assert out["n_known_ships"] == 6 and out["n_unknown_ships"] == 3


def test_open_set_scores_a_known_ship_counts_only_if_accepted_and_right():
    # known ships of class a: one right and confident, one right but unsure (rejected), one confident but wrong
    probs = np.vstack([_onehot([0], conf=0.95), _onehot([0], conf=0.40), _onehot([1], conf=0.95),
                       _onehot([1], conf=0.95), _onehot([2], conf=0.95)])
    y = np.array([0, 0, 0, 1, 2])
    ships = np.array(["s1", "s2", "s3", "s4", "s5"])
    unknown = np.vstack([_onehot([2], conf=0.95), np.full((1, 3), 1 / 3)])     # one fooled, one caught
    out = U.open_set_scores(val_probs=_onehot([0, 1, 2], conf=0.95), val_ships=np.array(["v1", "v2", "v3"]),
                            known_probs=probs, known_y=y, known_ships=ships,
                            unknown_probs=unknown, unknown_ships=np.array(["u1", "u2"]),
                            unknown_types=np.array(["pilotvessel", "pilotvessel"]), names=NAMES, keep=1.0)
    assert out["threshold_ship"] == pytest.approx(0.95)
    assert out["open_set_per_ship_recall"]["a"] == pytest.approx(1 / 3)
    assert out["open_set_per_ship_recall"]["unknown"] == 0.5
    assert out["known_ships_rejected_by_class"]["a"] == pytest.approx(1 / 3)
    assert out["unknown_ships_called_as"] == {"a": 0.0, "b": 0.0, "c": 0.5}      # share of all unknown ships accepted as each class
    assert out["closed_per_ship_balanced_accuracy"] == pytest.approx((2 / 3 + 1 + 1) / 3)


def test_reference_is_lesson_21_gradient_boosting_scored_per_ship(tmp_path):
    table = _synthetic(tmp_path, np.random.default_rng(2))
    ref = U.reference(tmp_path, tmp_path / "out", folds=(1, 2))
    assert [r["fold"] for r in ref["results"]] == [1, 2]
    old = O.experiment_b(table, O.all_folds(table), O.load_engineered(tmp_path), 0, lambda m: None)
    for r, o in zip(ref["results"], old):
        assert r["auroc_ship"] == pytest.approx(o["auroc_ship"])            # the same model as lesson 21
        assert r["auroc_clip"] == pytest.approx(o["auroc_clip"])
    assert (tmp_path / "out" / "reference.json").exists()


def _fake_cnn_runs(tmp_path, table, eng_cols, n_frames=48):
    """A small random-weight fusion network per fold, saved like lesson 17's best.pt, and a fake mel cache."""
    rng = np.random.default_rng(0)
    np.save(tmp_path / "mel.npy", rng.normal(-60, 5, size=(len(table), 128, n_frames)).astype(np.float16))
    torch.manual_seed(0)
    for k in range(1, S.N_FOLDS + 1):
        model = FN.FusionNet(len(eng_cols), len(U.NAMES), pretrained=False)
        (tmp_path / "cnn" / f"f{k}_16k").mkdir(parents=True)
        save_checkpoint(tmp_path / "cnn" / f"f{k}_16k" / "best.pt", model=model.state_dict(), epoch=1, n_bands=128,
                        columns=list(eng_cols), eng_mean=[0.0] * len(eng_cols), eng_std=[1.0] * len(eng_cols),
                        norm={"mean": [-60.0] * 128, "std": [5.0] * 128})


def test_run_scores_every_left_out_clip_once_with_lesson_17_models(tmp_path):
    table = _synthetic(tmp_path, np.random.default_rng(0))
    eng = O.load_engineered(tmp_path)
    _fake_cnn_runs(tmp_path, table, eng.columns)
    out = tmp_path / "out"
    summary = U.run(tmp_path, tmp_path / "cnn", out, device="cpu", batch_size=16)

    assert len(summary["results"]) == S.N_FOLDS
    preds = pd.read_parquet(out / "predictions.parquet")
    left = np.flatnonzero(table.label.isin(O.LEFT_OUT).to_numpy())
    unknown = preds[preds.split == "unknown"]
    assert sorted(unknown.row) == sorted(left)                          # every left-out clip, exactly once
    known6 = np.flatnonzero(table.label.isin(S.CLASSES6).to_numpy())
    assert sorted(preds[preds.split == "test"].row) == sorted(known6)   # every known clip tested once
    assert not set(preds[preds.split == "unknown"].label) & set(S.CLASSES6)
    np.testing.assert_allclose(preds[[f"p_{n}" for n in U.NAMES]].sum(axis=1), 1, atol=1e-3)

    # the ship threshold comes from the validation ships of the same fold only
    r = summary["results"][0]
    v = preds[(preds.fold == 1) & (preds.split == "validation")]
    conf = O.per_ship_max(v[[f"p_{n}" for n in U.NAMES]].to_numpy(), v.ship.to_numpy())
    assert r["threshold_ship"] == pytest.approx(U.threshold(conf.to_numpy()))
    assert json.loads((out / "results.json").read_text())["keep_known"] == U.KEEP_KNOWN
