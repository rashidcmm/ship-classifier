"""Shared test fixtures: a tiny synthetic copy of the real data layout (data/mel.npy + data/clips.parquet).

120 clips = 12 classes x 10 clips, 128 mel bands x 64 frames. Each ship class has 4 ships:
A and B (4 clips each) are train, C (1 clip) validation, D (1 clip) test, so splits are
vessel-disjoint and every class appears in every split. Each class gets a distinct offset in its
own block of mel bands, so a small CNN can learn the classes quickly.
"""
import numpy as np
import pandas as pd
import pytest

from labels import CLASSES

N_FRAMES = 64


def make_mini_data(data_dir, n_frames=N_FRAMES, seed=0):
    rng = np.random.default_rng(seed)
    rows, mels = [], []
    mmsi_next = 100_000_000
    for ci, label in enumerate(CLASSES):
        if label == "background":
            roles = ["train"] * 8 + ["validation", "test"]
            ships = [0] * 10
            hydro = [None] * 10
        else:
            ids = [mmsi_next + k for k in range(4)]
            mmsi_next += 4
            ships = [ids[0]] * 4 + [ids[1]] * 4 + [ids[2], ids[3]]
            roles = ["train"] * 8 + ["validation", "test"]
            # ships A, C on hydrophone 2523; B, D on 2556
            hydro = (["ICLISTENAF2523"] * 4 + ["ICLISTENAF2556"] * 4
                     + ["ICLISTENAF2523", "ICLISTENAF2556"])
        for k in range(10):
            is_ship = label != "background"
            rows.append({
                "label": label,
                "path": f"../audio/{label}/clip_{ci:02d}_{k:02d}.wav",
                "MMSI": float(ships[k]),
                "hydrophone": hydro[k],
                "mean_distance": float(200 + 150 * k) if is_ship else np.nan,
                "mean_sog": float(2 + ci + 0.5 * k) if is_ship else np.nan,
                "split_f1": roles[k],
            })
            m = rng.normal(-40.0, 1.0, size=(128, n_frames))
            m[ci * 10:(ci + 1) * 10] += 8.0                     # class signature band
            mels.append(m)
    table = pd.DataFrame(rows)
    table.loc[(table.label == "cargo") & (table.index % 10 == 1), "mean_distance"] = np.nan  # one missing distance
    for k in range(2, 6):                                         # other folds: swap validation/test roles on odd folds
        swap = {"validation": "test", "test": "validation", "train": "train"}
        table[f"split_f{k}"] = table.split_f1.map(swap) if k % 2 else table.split_f1
    table["mel_row"] = np.arange(len(table))
    table["hydrophone"] = table.hydrophone.astype("string")
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "mel.npy", np.stack(mels).astype(np.float16))
    table.to_parquet(data_dir / "clips.parquet", index=False)
    return data_dir


@pytest.fixture
def mini_data(tmp_path):
    return make_mini_data(tmp_path / "data")
