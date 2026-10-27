"""Step A: compute the log-mel spectrogram of every clip once and store them in one file.

Outputs (in --out, default <project>/data):
  mel.npy        float16 array (clips, 128, 501), read later with memory-mapping (no need to fit in RAM)
  clips.parquet  one row per clip, same order as mel.npy: all 42 metadata columns + fold splits
                 (split_f1..split_f5 = train / validation / test) + audio checks
  features.json  feature parameters and a summary of the run

    python src/prepare_features.py                 # all 43,208 clips
    python src/prepare_features.py --limit 300     # quick test
"""
import argparse
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
from keepawake import keep_awake
from paths import AUDIO_ZIP, DATA_DIR
from shipzip import SplitZip

ROOT = "allclass_5hours"


def build_clip_table(z):
    """Read the 15 fold CSVs from the zip; one row per clip with its split in each of the 5 folds."""
    table = None
    for k in range(1, 6):
        parts = []
        for split in ("train", "validation", "test"):
            df = pd.read_csv(io.BytesIO(z.read(f"{ROOT}/fold_{k}/metadata_5s_{split}.csv")), low_memory=False)
            df[f"split_f{k}"] = split
            parts.append(df)
        fold = pd.concat(parts, ignore_index=True)
        if fold.path.duplicated().any():
            raise ValueError(f"fold {k}: a clip appears in more than one split")
        if table is None:
            table = fold
        else:
            if len(fold) != len(table) or set(fold.path) != set(table.path):
                raise ValueError(f"fold {k} does not contain the same clips as fold 1")
            table = table.merge(fold[["path", f"split_f{k}"]], on="path", how="left")
    table["zip_name"] = ROOT + "/" + table.path.str.replace(r"^\.\./", "", regex=True)
    missing = ~table.zip_name.isin(z.entries)
    if missing.any():
        raise FileNotFoundError(f"{missing.sum()} clips listed in the CSVs are not in the zip, e.g. {table.zip_name[missing].iloc[0]}")
    return table


def load_clip(z, name):
    wave, sr, subtype = F.decode_wav(z.read(name))
    n = wave.size
    wave = F.fix_length(F.resample(wave, sr))
    rms_db = float(10 * np.log10(np.mean(wave.astype(np.float64) ** 2) + 1e-20))
    return wave, sr, subtype, n, rms_db


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default=str(AUDIO_ZIP), help="last part of the split audio zip")
    ap.add_argument("--out", default=str(DATA_DIR))
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (testing)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6, help="threads reading/decoding WAVs")
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    z = SplitZip(args.zip)
    table = build_clip_table(z)
    # archive order = sequential disk reads = fastest; this order is also the row order of mel.npy
    table = table.iloc[sorted(range(len(table)), key=lambda i: z.sort_key(table.zip_name.iat[i]))].reset_index(drop=True)
    if args.limit:
        table = table.iloc[:args.limit].copy()
    n = len(table)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logmel = F.LogMel().to(device)
    mel = np.lib.format.open_memmap(out / "mel.npy.partial", mode="w+", dtype=np.float16,
                                    shape=(n, F.N_MELS, F.N_FRAMES))
    sr_l, subtype_l, nsamp_l, rms_l = [None] * n, [None] * n, np.zeros(n, np.int64), np.zeros(n, np.float32)

    print(f"{n} clips -> {out / 'mel.npy'}  ({n * F.N_MELS * F.N_FRAMES * 2 / 1e9:.2f} GB), computing on {device}")
    chunk = args.batch * 4
    with ThreadPoolExecutor(args.workers) as pool, tqdm(total=n, unit="clip", smoothing=0.05) as bar:
        for start in range(0, n, chunk):
            idx = range(start, min(start + chunk, n))
            loaded = list(pool.map(lambda i: load_clip(z, table.zip_name.iat[i]), idx))
            for b in range(0, len(loaded), args.batch):
                part = loaded[b:b + args.batch]
                wave = torch.from_numpy(np.stack([p[0] for p in part])).to(device)
                spec = logmel(wave).cpu().numpy().astype(np.float16)
                rows = list(idx)[b:b + len(part)]
                mel[rows[0]:rows[-1] + 1] = spec
                for r, p in zip(rows, part):
                    _, sr_l[r], subtype_l[r], nsamp_l[r], rms_l[r] = p
                bar.update(len(part))
    mel.flush()
    del mel

    table["source_sr"], table["source_subtype"] = sr_l, subtype_l
    table["source_samples"], table["rms_db"] = nsamp_l, rms_l
    table["mel_row"] = np.arange(n)
    for c in table.columns:                      # parquet needs one type per column
        if table[c].dtype == object:
            table[c] = table[c].astype("string")
    table.to_parquet(out / "clips.parquet.partial", index=False)

    summary = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "clips": n,
        "source_zip": args.zip,
        "params": {"sr": F.SR, "clip_seconds": F.CLIP_SECONDS, "n_fft": F.N_FFT, "hop": F.HOP,
                   "n_mels": F.N_MELS, "fmin": F.FMIN, "fmax": F.FMAX, "n_frames": F.N_FRAMES,
                   "power_floor": F.POWER_FLOOR, "scale": "10*log10(power), no per-clip normalisation",
                   "short_clips": "padded by repetition (wrap)"},
        "source_sample_rates": {str(k): int(v) for k, v in pd.Series(sr_l).value_counts().items()},
        "source_formats": {str(k): int(v) for k, v in pd.Series(subtype_l).value_counts().items()},
        "short_clips_padded": int((nsamp_l / np.array(sr_l, dtype=np.float64) < F.CLIP_SECONDS).sum()),
        "rms_db": {"min": float(rms_l.min()), "median": float(np.median(rms_l)), "max": float(rms_l.max())},
        "seconds": round(time.time() - t0, 1),
    }
    for name in ("mel.npy", "clips.parquet"):    # only now replace any previous complete output
        (out / name).unlink(missing_ok=True)
        (out / f"{name}.partial").rename(out / name)
    (out / "features.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    with keep_awake():
        main()
