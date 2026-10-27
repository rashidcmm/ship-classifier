"""Which clips go into each experiment, and how one clip becomes a model input.

The features (data/mel.npy) hold one 128 x 501 log-mel spectrogram per clip; data/clips.parquet holds
one row per clip in the same order, with the label, targets and the clip's split in each of 5 folds.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from labels import load_groups

HYDROPHONE_TRAIN, HYDROPHONE_TEST = "ICLISTENAF2523", "ICLISTENAF2556"
SPLITS = ("train", "validation", "test")


class DataError(RuntimeError):
    """The feature files are missing or damaged; the message says which file to copy again."""


def open_features(data_dir: Path) -> tuple[np.memmap, pd.DataFrame]:
    data_dir = Path(data_dir)
    mel_path, table_path = data_dir / "mel.npy", data_dir / "clips.parquet"
    for p in (mel_path, table_path):
        if not p.exists():
            raise DataError(f"{p.name} not found in {data_dir}. Copy the whole data folder from the "
                            f"preparation laptop (or run src/prepare_features.py).")
    try:
        mel = np.load(mel_path, mmap_mode="r")
    except (ValueError, OSError, EOFError) as e:
        raise DataError(f"mel.npy in {data_dir} is damaged or incompletely copied ({e}). Copy it again.") from e
    try:
        table = pd.read_parquet(table_path)
    except Exception as e:
        raise DataError(f"clips.parquet in {data_dir} cannot be read ({e}). Copy it again.") from e
    if mel.ndim != 3 or len(mel) != len(table):
        raise DataError(f"mel.npy has {len(mel)} clips but clips.parquet has {len(table)} rows: "
                        f"the two files come from different runs. Copy both again.")
    return mel, table


def select_rows(table: pd.DataFrame, *, fold: int, split: str, task: str, holdout: str | None = None,
                split_mode: str = "vessel", hydrophone_shift: bool = False, seed: int = 0) -> np.ndarray:
    """Row indices for one split of one experiment (rules: spec sections 3 and 5)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    col = table[f"split_f{fold}"].to_numpy()
    if split_mode == "vessel":
        keep = col == split
    elif split_mode == "leaky":
        # same split sizes as the vessel split, but clips are shuffled without regard to which ship they come from
        counts = [int((col == s).sum()) for s in SPLITS]
        perm = np.random.default_rng(seed).permutation(len(table))
        bounds = np.cumsum([0] + counts)
        i = SPLITS.index(split)
        keep = np.zeros(len(table), bool)
        keep[perm[bounds[i]:bounds[i + 1]]] = True
    else:
        raise ValueError("split_mode must be 'vessel' or 'leaky'")

    label = table.label.to_numpy()
    if holdout is not None and split != "test":
        keep &= ~np.isin(label, load_groups()[holdout])
    if task in ("distance", "speed"):
        target = table["mean_distance" if task == "distance" else "mean_sog"].to_numpy(dtype=float)
        keep &= (label != "background") & np.isfinite(target)
    elif task not in ("class", "multi"):
        raise ValueError(f"unknown task {task!r}")
    if hydrophone_shift:
        hyd = table.hydrophone.astype("string").fillna("").to_numpy()
        wanted = HYDROPHONE_TEST if split == "test" else HYDROPHONE_TRAIN
        keep &= (label == "background") | (hyd == wanted)
    return np.flatnonzero(keep)


def merge_into_other(table: pd.DataFrame, classes) -> pd.DataFrame:
    """A copy of the table with these classes relabelled "other" (classes with too few ships to learn on their own)."""
    table = table.copy()
    table["label"] = table.label.where(~table.label.isin(list(classes)), "other")
    return table


def compute_norm(mel, rows, max_clips: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Per-band mean and std of the training clips (never the test clips), used to normalise every input."""
    rows = np.asarray(rows)
    if len(rows) > max_clips:
        rows = np.sort(np.random.default_rng(seed).choice(rows, max_clips, replace=False))
    total = np.zeros(mel.shape[1]); total_sq = np.zeros(mel.shape[1]); n = 0
    for start in range(0, len(rows), 200):
        block = np.asarray(mel[rows[start:start + 200]], dtype=np.float64)   # (b, bands, frames)
        total += block.sum(axis=(0, 2)); total_sq += (block ** 2).sum(axis=(0, 2))
        n += block.shape[0] * block.shape[2]
    mean = total / n
    std = np.sqrt(np.maximum(total_sq / n - mean ** 2, 0.0))
    return mean.astype(np.float32), np.maximum(std, 1e-3).astype(np.float32)


def spec_augment(x: torch.Tensor, cfg: dict, generator: torch.Generator | None = None) -> torch.Tensor:
    """SpecAugment: hide random frequency bands and time spans (set to 0 = the average after normalisation)."""
    _, n_bands, n_frames = x.shape
    for _ in range(cfg["freq_masks"]):
        w = int(torch.randint(0, cfg["freq_width"] + 1, (1,), generator=generator))
        f0 = int(torch.randint(0, max(1, n_bands - w), (1,), generator=generator))
        x[:, f0:f0 + w, :] = 0.0
    for _ in range(cfg["time_masks"]):
        w = int(torch.randint(0, min(cfg["time_width"], n_frames) + 1, (1,), generator=generator))
        t0 = int(torch.randint(0, max(1, n_frames - w), (1,), generator=generator))
        x[:, :, t0:t0 + w] = 0.0
    return x


DEFAULT_AUG = {"freq_masks": 2, "freq_width": 16, "time_masks": 2, "time_width": 50}


class MelDataset(torch.utils.data.Dataset):
    """One item = one clip: normalised spectrogram + class index (-1 = unknown class) + distance/speed targets."""

    def __init__(self, data_dir: Path, table: pd.DataFrame, rows: np.ndarray, known: list[str],
                 norm: tuple, augment: bool, aug_cfg: dict | None = None):
        self.mel_path = Path(data_dir) / "mel.npy"
        self.rows = np.asarray(rows, dtype=np.int64)
        index = {c: i for i, c in enumerate(known)}
        self.y_class = np.array([index.get(c, -1) for c in table.label.to_numpy()], dtype=np.int64)
        self.y_dist = (table.mean_distance.to_numpy(dtype=np.float64) / 1000.0).astype(np.float32)   # km
        self.y_speed = (table.mean_sog.to_numpy(dtype=np.float64) / 10.0).astype(np.float32)        # knots / 10
        self.mean = torch.from_numpy(np.asarray(norm[0], np.float32))[:, None]
        self.std = torch.from_numpy(np.asarray(norm[1], np.float32))[:, None]
        self.augment, self.aug_cfg = augment, aug_cfg or DEFAULT_AUG
        self._mel = None                      # opened lazily in each worker process

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mel"] = None
        return state

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self._mel is None:
            self._mel = np.load(self.mel_path, mmap_mode="r")
        r = int(self.rows[i])
        x = (torch.from_numpy(np.asarray(self._mel[r], dtype=np.float32)) - self.mean) / self.std
        x = x.unsqueeze(0)
        if self.augment:
            x = spec_augment(x, self.aug_cfg)
        return {"x": x, "y_class": torch.tensor(self.y_class[r]), "y_dist": torch.tensor(self.y_dist[r]),
                "y_speed": torch.tensor(self.y_speed[r]), "row": torch.tensor(r)}
