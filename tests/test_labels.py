import numpy as np
import pandas as pd
import pytest

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


def test_groupings_valid():
    groupings = labels.load_groupings()
    assert sorted(groupings) == ["shipsear5", "uniqueship5"]
    for groups in groupings.values():
        members = [c for m in groups.values() for c in m]
        assert len(groups) == 5 and len(members) == len(set(members)) and set(members) <= set(labels.CLASSES)
    size = groupings["shipsear5"]
    assert list(size) == ["working", "small", "passenger", "large", "background"]
    assert size["large"] == ["cargo", "tanker"] and size["background"] == ["background"]
    assert "other" not in {c for m in size.values() for c in m}           # fits no size group: left out
    assert list(groupings["uniqueship5"]) == ["background", "cargo", "passengership", "tanker", "tug"]


def test_load_groupings_rejects_bad_files(tmp_path):
    bad = tmp_path / "groups.yaml"
    bad.write_text("g: {a: [cargo], b: [cargo, tug]}\n")
    with pytest.raises(ValueError, match="more than one group"):
        labels.load_groupings(bad)
    bad.write_text("g: {a: [boat]}\n")
    with pytest.raises(ValueError, match="not a class name"):
        labels.load_groupings(bad)


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
