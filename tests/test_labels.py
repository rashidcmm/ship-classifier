import numpy as np
import pandas as pd

import labels


def test_classes_fixed_order():
    assert len(labels.CLASSES) == 12
    assert labels.CLASSES[0] == "background"
    assert labels.CLASSES[-1] == "tug"


def test_groups_disjoint_and_valid():
    groups = labels.load_groups()
    assert sorted(groups) == ["A", "B", "C"]
    members = [set(v) for v in groups.values()]
    assert all(len(m) == 3 for m in members)
    assert not (members[0] & members[1] or members[0] & members[2] or members[1] & members[2])
    never = {"background", "pilotvessel", "dredger"}
    assert all(not (m & never) for m in members)
    assert all(m <= set(labels.CLASSES) for m in members)


def test_known_classes_holdout_A():
    known = labels.known_classes("A")
    assert len(known) == 9
    assert not {"fishing", "sailing", "tanker"} & set(known)
    assert known == [c for c in labels.CLASSES if c in known]
    assert labels.known_classes(None) == labels.CLASSES


def test_load_config_defaults():
    cfg = labels.load_config()
    assert cfg["backbone"] == "efficientnet_b0"
    assert cfg["epochs"]["class"] == 25
    assert cfg["specaugment"]["time_width"] == 50


def test_mini_data_fixture(mini_data):
    mel = np.load(mini_data / "mel.npy", mmap_mode="r")
    table = pd.read_parquet(mini_data / "clips.parquet")
    assert len(table) == 120
    assert mel.shape == (120, 128, 64)
    for split in ("train", "validation", "test"):
        assert set(table.label[table.split_f1 == split]) == set(labels.CLASSES)
