import json

import benchmark
import check_setup


def test_check_setup_ok(mini_data, capsys):
    assert check_setup.main(["--data-dir", str(mini_data), "--write"]) == 0
    sums = json.loads((mini_data / "checksums.json").read_text())
    assert set(sums) == {"clips.parquet", "mel.npy"} and "size" in sums["mel.npy"]
    assert check_setup.main(["--data-dir", str(mini_data)]) == 0
    assert "OK" in capsys.readouterr().out


def test_check_setup_truncated_mel(mini_data, capsys):
    check_setup.main(["--data-dir", str(mini_data), "--write"])
    raw = (mini_data / "mel.npy").read_bytes()
    (mini_data / "mel.npy").write_bytes(raw[: len(raw) // 2])
    capsys.readouterr()
    assert check_setup.main(["--data-dir", str(mini_data)]) != 0
    assert "mel.npy" in capsys.readouterr().out


def test_check_setup_changed_parquet(mini_data, capsys):
    check_setup.main(["--data-dir", str(mini_data), "--write"])
    with open(mini_data / "clips.parquet", "ab") as f:
        f.write(b"x")
    capsys.readouterr()
    assert check_setup.main(["--data-dir", str(mini_data)]) != 0
    assert "clips.parquet" in capsys.readouterr().out


def test_benchmark_returns_estimates(mini_data):
    res = benchmark.main(["--data-dir", str(mini_data), "--backbone", "tiny_cnn", "--steps", "3",
                          "--no-pretrained", "--device", "cpu", "--workers", "0"])
    assert res["E2"] > 0 and res["total_hours"] >= res["E2"]
    assert res["train_clips_per_s"] > 0 and res["batch_size"] == 16


def test_check_setup_device_failure_message(mini_data, monkeypatch, capsys):
    def broken(dev):
        raise RuntimeError("CUDA error: no kernel image is available for execution on the device")
    monkeypatch.setattr(check_setup, "device_smoke_test", broken)
    assert check_setup.main(["--data-dir", str(mini_data)]) != 0
    out = capsys.readouterr().out
    assert "no kernel image" in out and "README" in out


def test_benchmark_and_baseline_missing_data_messages(tmp_path):
    import baseline_classical
    import pytest
    for main, argv in ((benchmark.main, ["--data-dir", str(tmp_path), "--backbone", "tiny_cnn", "--no-pretrained"]),
                       (baseline_classical.main, ["--data-dir", str(tmp_path), "--runs-dir", str(tmp_path / "r")])):
        with pytest.raises(SystemExit) as e:
            main(argv)
        assert "mel.npy" in str(e.value.code) or "clips.parquet" in str(e.value.code)


def test_benchmark_counts_holdout_ensemble():
    runs, task, extra = benchmark.EXPERIMENTS["ensemble"]
    assert runs == 2 and task == "class" and extra.get("holdout") == "A"
