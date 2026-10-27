import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import evaluate
import predict
import train
from conftest import make_mini_data


@pytest.fixture(scope="module")
def class_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("pred")
    mini = make_mini_data(root / "data")
    argv = ["--data-dir", str(mini), "--runs-dir", str(root / "runs"), "--backbone", "tiny_cnn",
            "--no-pretrained", "--workers", "0", "--device", "cpu", "--task", "class", "--epochs", "1"]
    rd = train.run(train.make_parser().parse_args(argv))
    dist = train.run(train.make_parser().parse_args([*argv[:-4], "--task", "distance", "--epochs", "1"]))
    return rd, dist


def write_wav(path, seconds, sr, channels):
    t = np.arange(int(seconds * sr)) / sr
    mono = 0.05 * np.sin(2 * np.pi * 120 * t) + 0.01 * np.random.default_rng(0).normal(size=t.size)
    sf.write(path, np.stack([mono] * channels, axis=1) if channels > 1 else mono, sr, subtype="PCM_32")
    return path


def test_predict_odd_wavs(class_run, tmp_path):
    rd, dist = class_run
    df = predict.predict_file(write_wav(tmp_path / "long.wav", 13, 48_000, 2), rd, distance_run=dist)
    windows = df[df.start_s != "overall"]
    assert len(windows) == 3 and df.iloc[-1].start_s == "overall"    # 0-5, 5-10, 10-13 padded
    assert list(windows.end_s) == [5.0, 10.0, 13.0]
    assert {"class", "confidence", "unknown", "distance_m", "speed_kn"} <= set(df.columns)
    short = predict.predict_file(write_wav(tmp_path / "short.wav", 3, 16_000, 1), rd)
    assert len(short) == 2
    with pytest.raises(ValueError, match="2.5 s"):
        predict.predict_file(write_wav(tmp_path / "tiny.wav", 2, 32_000, 1), rd)


def test_predict_without_calibration_warns(class_run, tmp_path, capsys):
    rd, _ = class_run
    predict.predict_file(write_wav(tmp_path / "a.wav", 5, 32_000, 1), rd)
    assert "evaluate.py" in capsys.readouterr().out


def test_predict_uses_calibration_and_blanks_unknown(class_run, tmp_path):
    rd, dist = class_run
    evaluate.evaluate_run(rd, device="cpu")
    df = predict.predict_file(write_wav(tmp_path / "b.wav", 10, 32_000, 1), rd, distance_run=dist)
    w = df[df.start_s != "overall"]
    blank = w.unknown | (w["class"] == "background")
    assert w.loc[blank, "distance_m"].isna().all()
    assert w.loc[~blank, "distance_m"].notna().all()


def test_predict_chunked_matches_unchunked(class_run, tmp_path, monkeypatch):
    rd, dist = class_run
    wav = write_wav(tmp_path / "c.wav", 23, 32_000, 1)
    whole = predict.predict_file(wav, rd, distance_run=dist)
    monkeypatch.setattr(predict, "CHUNK", 2)
    chunked = predict.predict_file(wav, rd, distance_run=dist)
    pd.testing.assert_frame_equal(whole, chunked, check_exact=False, rtol=1e-4)
