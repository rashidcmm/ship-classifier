"""One pass over the audio: PANN embeddings for every clip, engineered features for the left-out classes (lesson 21).

  - PANNs (Kong et al., 2020): CNN14 pretrained on AudioSet (2 million YouTube clips, 527 sound types), 32 kHz input.
    Each 5-s clip -> its 2,048-number embedding (the layer before AudioSet's output), saved for all 12 classes.
  - The 144 engineered features of lesson 13 for the six classes we had left out (sailing, other, passenger ship,
    rescue, dredger, pilot vessel), so they can form an "other" class or serve as unknown ships.

  python src/panns_features.py      # -> data/panns_embeddings.npy (+ _rows.npy), data/features_leftout_engineered.parquet
Needs the audio zip, the GPU, and the CNN14 checkpoint in ~/panns_data (downloaded once).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import six_classes as S
from features import decode_wav, fix_length, resample
from keepawake import keep_awake
from paths import AUDIO_ZIP, DATA_DIR

EMB_FILE, EMB_ROWS_FILE = "panns_embeddings.npy", "panns_embeddings_rows.npy"
LEFTOUT_FILE = "features_leftout_engineered.parquet"


class ClipAudio(torch.utils.data.Dataset):
    """One item: (row, 5-s waveform at 32 kHz, engineered features or None)."""

    def __init__(self, table, rows, zip_path, engineered_rows):
        self.names = table.zip_name.to_numpy()
        self.rows, self.zip_path, self.eng_rows = np.asarray(rows), str(zip_path), set(map(int, engineered_rows))
        self._zip = None

    def __getstate__(self):
        return {**self.__dict__, "_zip": None}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self._zip is None:
            from shipzip import SplitZip
            self._zip = SplitZip(self.zip_path)
        r = int(self.rows[i])
        wave, sr, _ = decode_wav(self._zip.read(self.names[r]))
        wave = fix_length(resample(wave, sr)).astype(np.float32)
        eng = S.engineered_features(wave) if r in self.eng_rows else None
        return r, wave, eng


def collate(items):
    return [i[0] for i in items], np.stack([i[1] for i in items]), [i[2] for i in items]


def run(data_dir=DATA_DIR, zip_path=AUDIO_ZIP, workers=4, batch_size=32, limit=0, device="cuda"):
    from panns_inference import AudioTagging
    t0 = time.time()
    table = pd.read_parquet(Path(data_dir) / "clips.parquet")
    rows = np.flatnonzero(table.label.notna().to_numpy())
    if limit:
        rows = np.sort(np.random.default_rng(0).choice(rows, limit, replace=False))
    leftout = rows[~table.label.iloc[rows].isin(S.CLASSES6).to_numpy()]
    ds = ClipAudio(table, rows, zip_path, leftout)
    loader = torch.utils.data.DataLoader(ds, batch_size, shuffle=False, num_workers=workers, collate_fn=collate,
                                         persistent_workers=workers > 0)
    model = AudioTagging(checkpoint_path=None, device=device)
    emb, emb_rows, eng = [], [], []
    for b, (r, waves, engs) in enumerate(loader):
        _, e = model.inference(waves)
        emb.append(e.astype(np.float16)); emb_rows += r
        eng += [{"row": ri, **ei} for ri, ei in zip(r, engs) if ei is not None]
        if b % 50 == 0:
            print(f"[{(time.time() - t0) / 60:5.1f} min] {len(emb_rows):,} / {len(rows):,} clips", flush=True)
    np.save(Path(data_dir) / EMB_FILE, np.concatenate(emb))
    np.save(Path(data_dir) / EMB_ROWS_FILE, np.asarray(emb_rows))
    pd.DataFrame(eng).astype({"row": int}).to_parquet(Path(data_dir) / LEFTOUT_FILE)
    print(f"done: {len(emb_rows):,} embeddings, {len(eng):,} left-out clips with engineered features, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


def load_embeddings(data_dir=DATA_DIR) -> pd.DataFrame:
    """PANN embeddings as a table indexed by clip row (columns pann_0 ... pann_2047)."""
    e = np.load(Path(data_dir) / EMB_FILE).astype(np.float32)
    rows = np.load(Path(data_dir) / EMB_ROWS_FILE)
    return pd.DataFrame(e, index=rows, columns=[f"pann_{i}" for i in range(e.shape[1])])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="only N random clips (quick test)")
    a = p.parse_args(argv)
    with keep_awake():
        run(a.data_dir, workers=a.workers, limit=a.limit)


if __name__ == "__main__":
    main()
