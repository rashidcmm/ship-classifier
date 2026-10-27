import numpy as np
import pandas as pd
import pytest
import torch

import data
from labels import known_classes


@pytest.fixture
def feats(mini_data):
    mel, table = data.open_features(mini_data)
    return mini_data, mel, table


def test_vessel_split_matches_column(feats):
    _, _, table = feats
    rows = data.select_rows(table, fold=1, split="train", task="class")
    assert np.array_equal(rows, np.flatnonzero(table.split_f1.to_numpy() == "train"))


def test_vessel_split_no_shared_mmsi(feats):
    _, _, table = feats
    tr = data.select_rows(table, fold=1, split="train", task="class")
    te = data.select_rows(table, fold=1, split="test", task="class")
    ships = lambda r: set(table.MMSI.iloc[r][table.MMSI.iloc[r] > 0])
    assert not ships(tr) & ships(te)


def test_leaky_split_counts_and_overlap(feats):
    _, _, table = feats
    for split in ("train", "validation", "test"):
        v = data.select_rows(table, fold=1, split=split, task="class")
        lk = data.select_rows(table, fold=1, split=split, task="class", split_mode="leaky")
        assert len(lk) == len(v)
    tr = data.select_rows(table, fold=1, split="train", task="class", split_mode="leaky")
    te = data.select_rows(table, fold=1, split="test", task="class", split_mode="leaky")
    assert not set(tr) & set(te)
    ships = lambda r: set(table.MMSI.iloc[r][table.MMSI.iloc[r] > 0])
    assert ships(tr) & ships(te), "leaky split should put clips of one ship in both train and test"


def test_holdout_removes_from_train_keeps_test(feats):
    _, _, table = feats
    held = {"fishing", "sailing", "tanker"}
    for split in ("train", "validation"):
        r = data.select_rows(table, fold=1, split=split, task="class", holdout="A")
        assert not held & set(table.label.iloc[r])
    te = data.select_rows(table, fold=1, split="test", task="class", holdout="A")
    assert held <= set(table.label.iloc[te])


def test_regression_task_drops_background_and_nan(feats):
    _, _, table = feats
    r = data.select_rows(table, fold=1, split="train", task="distance")
    sub = table.iloc[r]
    assert "background" not in set(sub.label)
    assert sub.mean_distance.notna().all()
    ships_train = ((table.split_f1 == "train") & (table.label != "background")).sum()
    assert len(r) == ships_train - 1                      # the one missing cargo distance
    s = data.select_rows(table, fold=1, split="train", task="speed")
    assert len(s) == ships_train


def test_hydrophone_shift(feats):
    _, _, table = feats
    tr = data.select_rows(table, fold=1, split="train", task="class", hydrophone_shift=True)
    te = data.select_rows(table, fold=1, split="test", task="class", hydrophone_shift=True)
    assert "ICLISTENAF2556" not in set(table.hydrophone.iloc[tr].dropna())
    assert "ICLISTENAF2523" not in set(table.hydrophone.iloc[te].dropna())
    assert "background" in set(table.label.iloc[tr]) and "background" in set(table.label.iloc[te])


def test_norm_shapes_and_floor(feats):
    data_dir, mel, table = feats
    rows = data.select_rows(table, fold=1, split="train", task="class")
    mean, std = data.compute_norm(mel, rows)
    assert mean.shape == (128,) and std.shape == (128,)
    assert mean.dtype == np.float32 and std.dtype == np.float32
    assert (std >= 1e-3).all()
    const = np.full((3, 128, 8), -50.0, dtype=np.float16)
    _, s = data.compute_norm(const, np.arange(3))
    assert np.allclose(s, 1e-3)


def test_dataset_item(feats):
    data_dir, mel, table = feats
    rows = np.arange(len(table))
    norm = data.compute_norm(mel, rows)
    ds = data.MelDataset(data_dir, table, rows, known=known_classes("A"), norm=norm, augment=False)
    fishing = int(np.flatnonzero(table.label.to_numpy() == "fishing")[0])
    item = ds[fishing]
    assert item["x"].shape == (1, 128, 64) and item["x"].dtype == torch.float32
    assert item["y_class"] == -1
    bg = ds[int(np.flatnonzero(table.label.to_numpy() == "background")[0])]
    assert torch.isnan(bg["y_dist"]) and bg["y_class"] == known_classes("A").index("background")
    tug = int(np.flatnonzero(table.label.to_numpy() == "tug")[3])
    assert ds[tug]["y_speed"].item() == pytest.approx(table.mean_sog.iloc[tug] / 10, rel=1e-5)
    assert ds[tug]["y_dist"].item() == pytest.approx(table.mean_distance.iloc[tug] / 1000, rel=1e-5)
    assert ds[tug]["row"] == tug


def test_dataset_in_worker_processes(feats):
    data_dir, mel, table = feats
    rows = np.arange(len(table))
    ds = data.MelDataset(data_dir, table, rows, known=known_classes(None),
                         norm=data.compute_norm(mel, rows), augment=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=2)
    n = sum(len(b["row"]) for b in loader)
    assert n == len(table)


def test_spec_augment_masks():
    x = torch.ones(1, 128, 200)
    cfg = {"freq_masks": 2, "freq_width": 16, "time_masks": 2, "time_width": 50}
    g = torch.Generator().manual_seed(0)
    y = data.spec_augment(x.clone(), cfg, generator=g)
    assert y.shape == x.shape
    assert (y == 0).any() and (y == 1).any()
    assert (y == 0).float().mean() < 0.5


def test_open_features_truncated(mini_data):
    raw = (mini_data / "mel.npy").read_bytes()
    (mini_data / "mel.npy").write_bytes(raw[:1000])
    with pytest.raises(data.DataError, match="mel.npy"):
        data.open_features(mini_data)


def test_open_features_missing(tmp_path):
    with pytest.raises(data.DataError, match="clips.parquet|mel.npy"):
        data.open_features(tmp_path)
