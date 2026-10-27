import json

import numpy as np
import pandas as pd
import pytest
import torch

import data
import train
from models import ShipNet, save_checkpoint
from labels import CLASSES


def run_train(mini_data, tmp_path, *extra):
    argv = ["--data-dir", str(mini_data), "--runs-dir", str(tmp_path / "runs"), "--backbone", "tiny_cnn",
            "--no-pretrained", "--workers", "0", "--device", "cpu", *extra]
    return train.run(train.make_parser().parse_args(argv))


def log(run_dir):
    return pd.read_csv(run_dir / "log.csv")


def test_train_class_writes_run(mini_data, tmp_path):
    rd = run_train(mini_data, tmp_path, "--task", "class", "--epochs", "2")
    assert rd.name == "class_f1_tiny_cnn_vessel"
    assert len(log(rd)) == 2
    for f in ("best.pt", "last.pt", "norm.npz", "done.json", "config.json"):
        assert (rd / f).exists(), f
    cfg = json.loads((rd / "config.json").read_text())
    assert cfg["known"] == CLASSES and cfg["batch_size"] == 16 and cfg["n_train"] == 96


def test_train_learns_mini(mini_data, tmp_path):
    rd = run_train(mini_data, tmp_path, "--task", "class", "--epochs", "15", "--lr", "3e-3")
    assert log(rd).val_metric.max() > 0.5


def test_train_config_file_overrides_defaults(mini_data, tmp_path):
    extra = tmp_path / "strong.yaml"
    extra.write_text("lr: 1.0e-4\ndrop_rate: 0.5\nspecaugment: {freq_masks: 3, freq_width: 24, time_masks: 3, time_width: 20}\n")
    rd = run_train(mini_data, tmp_path, "--task", "class", "--epochs", "1", "--config", str(extra))
    cfg = json.loads((rd / "config.json").read_text())
    assert cfg["lr"] == 1e-4 and cfg["drop_rate"] == 0.5 and cfg["specaugment"]["time_width"] == 20
    assert cfg["weight_decay"] == 0.01                 # not in the file: default.yaml value


def test_train_distance_ignores_background(mini_data, tmp_path):
    rd = run_train(mini_data, tmp_path, "--task", "distance", "--epochs", "1")
    cfg = json.loads((rd / "config.json").read_text())
    _, table = data.open_features(mini_data)
    rows = data.select_rows(table, fold=1, split="train", task="distance")
    assert cfg["n_train"] == len(rows)
    assert "background" not in set(table.label.iloc[rows])
    assert np.isfinite(log(rd).val_metric).all() and (log(rd).val_metric > 1).all()   # MAE in metres


def test_train_multi_task_runs(mini_data, tmp_path):
    rd = run_train(mini_data, tmp_path, "--task", "multi", "--epochs", "1")
    assert np.isfinite(log(rd).train_loss).all()


def test_holdout_known_classes(mini_data, tmp_path):
    rd = run_train(mini_data, tmp_path, "--task", "class", "--epochs", "1", "--holdout", "A")
    assert rd.name == "class_f1_tiny_cnn_vessel_hA"
    cfg = json.loads((rd / "config.json").read_text())
    assert len(cfg["known"]) == 9 and "fishing" not in cfg["known"]


def test_resume_after_kill(mini_data, tmp_path):
    rd = run_train(mini_data, tmp_path, "--task", "class", "--epochs", "3", "--max-hours", "0")
    assert log(rd).epoch.tolist() == [1]
    assert not (rd / "done.json").exists()
    (rd / "last.pt.tmp").write_bytes(b"half-written garbage from a power cut")
    rd2 = run_train(mini_data, tmp_path, "--task", "class", "--epochs", "3", "--resume")
    assert rd2 == rd
    assert log(rd).epoch.tolist() == [1, 2, 3]
    assert (rd / "done.json").exists()


def test_existing_run_requires_resume(mini_data, tmp_path):
    run_train(mini_data, tmp_path, "--task", "class", "--epochs", "1")
    with pytest.raises(SystemExit, match="--resume"):
        run_train(mini_data, tmp_path, "--task", "class", "--epochs", "1")


def test_init_from_class_run(tmp_path):
    src = ShipNet("tiny_cnn", n_classes=12, pretrained=False)
    save_checkpoint(tmp_path / "best.pt", model=src.state_dict(), config={"backbone": "tiny_cnn"},
                    known=CLASSES, norm={"mean": [0.0] * 128, "std": [1.0] * 128})
    dst = ShipNet("tiny_cnn", n_classes=9, pretrained=False)
    skipped = train.load_init_weights(dst, tmp_path / "best.pt")
    assert torch.equal(dst.backbone.body[0].weight, src.backbone.body[0].weight)
    assert any(k.startswith("class_head") for k in skipped)


# ---------- review fixes ----------
def test_resume_loads_checkpoint_on_cpu(mini_data, tmp_path, monkeypatch):
    run_train(mini_data, tmp_path, "--task", "class", "--epochs", "2", "--max-hours", "0")
    seen = []
    real_load = torch.load

    def spy(*a, **k):
        seen.append(k.get("map_location"))
        return real_load(*a, **k)

    monkeypatch.setattr(train.torch, "load", spy)
    run_train(mini_data, tmp_path, "--task", "class", "--epochs", "2", "--resume")
    # must be the literal "cpu", independent of the training device: with map_location=<cuda device> the saved
    # RNG states land on the GPU and torch.set_rng_state crashes (only visible on a GPU machine)
    assert seen and all(m == "cpu" and isinstance(m, str) for m in seen)


def test_resume_finished_run_message(mini_data, tmp_path):
    run_train(mini_data, tmp_path, "--task", "class", "--epochs", "1")
    with pytest.raises(SystemExit, match="already finished"):
        run_train(mini_data, tmp_path, "--task", "class", "--epochs", "1", "--resume")


def test_main_missing_data_is_plain_message(tmp_path):
    with pytest.raises(SystemExit) as e:
        train.main(["--task", "class", "--data-dir", str(tmp_path / "nothing"), "--runs-dir", str(tmp_path / "r"),
                    "--backbone", "tiny_cnn", "--no-pretrained", "--device", "cpu"])
    assert "mel.npy" in str(e.value.code) or "clips.parquet" in str(e.value.code)


def test_init_from_refuses_other_fold(mini_data, tmp_path):
    src = run_train(mini_data, tmp_path, "--task", "class", "--epochs", "1")
    with pytest.raises(SystemExit, match="fold"):
        run_train(mini_data, tmp_path, "--task", "distance", "--epochs", "1", "--fold", "2",
                  "--init-from", str(src / "best.pt"))


def test_log_write_survives_locked_file(tmp_path, monkeypatch, capsys):
    path = tmp_path / "log.csv"
    calls = {"n": 0}
    real_open = open

    def locked_once(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("file is open in Excel")
        return real_open(*a, **k)

    monkeypatch.setattr(train, "open", locked_once, raising=False)
    row = {f: 1 for f in train.LOG_FIELDS}
    pending = train.append_log(path, [dict(row, epoch=1)])
    assert len(pending) == 1 and "log.csv" in capsys.readouterr().out
    pending = train.append_log(path, pending + [dict(row, epoch=2)])
    assert pending == [] and pd.read_csv(path).epoch.tolist() == [1, 2]
