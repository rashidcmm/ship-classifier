"""CNNs with the "other" class (lesson 25): which picture of the sound, and do the chosen features help.

Seven classes (6 + other), the folds, ship weights and per-ship scoring of lessons 22-24. One network for all runs:
EfficientNet-B0 (ImageNet start, one input channel), lesson 17's recipe (AdamW 3e-4, one warm-up epoch then cosine,
SpecAugment, dropout 0.3, at most 12 epochs, the epoch kept is the one with the best validation balanced accuracy,
stop after 3 epochs without improvement). Inputs:

  mel    the cached log-mel spectrogram: 128 bands x 501 frames (10 Hz - 16 kHz), "before MFCC"
  mfcc   the same frames after the DCT over the bands: 40 coefficients x 501 frames, "after MFCC"
  stft   a linear-frequency spectrogram computed from the audio: 1,024-sample window (31 Hz bins, 513 bins up to
         16 kHz) x 501 frames, the "full spectrogram" without any mel grouping

  --features   the feature sets kept by lesson 23's walk (runs/with_other/walk_result.json) join the CNN's 1,280
               numbers before the output layer, as in lesson 17

  python src/cnn_with_other.py --input mel --fold 1                                  # -> runs/cnn_with_other/f1_mel/
  python src/cnn_with_other.py --input mel --features --fold 1                       # -> runs/cnn_with_other/f1_mel_features/
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as TF
from scipy.fft import dct
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D
import features as F
import merged_classes as M
import open_set as O
import with_other as W
from device import TemperatureGuard, auto_batch_size, pick_device
from keepawake import keep_awake
from models import save_checkpoint
from paths import AUDIO_ZIP, DATA_DIR, RUNS_DIR

OUT_DIR = RUNS_DIR / "cnn_with_other"
INPUTS = ("mel", "mfcc", "stft")
N_MFCC_IMAGE = 40
STFT_FFT = 1024
SETTINGS = dict(epochs=12, patience=3, lr=3e-4, weight_decay=0.01, warmup_epochs=1, drop_rate=0.3, eng_hidden=128,
                specaugment={"freq_masks": 2, "freq_width": 16, "time_masks": 2, "time_width": 50})
NAMES = list(W.GROUPINGS[W.MAIN])


def input_rows(kind: str) -> int:
    return {"mel": F.N_MELS, "mfcc": N_MFCC_IMAGE, "stft": STFT_FFT // 2 + 1}[kind]


def stft_db(wave: np.ndarray) -> np.ndarray:
    """Linear-frequency power spectrogram in dB, (513 bins, 501 frames), the same 10 ms hop as the mel cache."""
    x = np.pad(wave.astype(np.float32), STFT_FFT // 2, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(x, STFT_FFT)[::F.HOP][:F.N_FRAMES]
    power = np.abs(np.fft.rfft(frames * np.hanning(STFT_FFT).astype(np.float32), axis=1)) ** 2
    return (10 * np.log10(power + F.POWER_FLOOR)).T.astype(np.float32)


class Pictures(torch.utils.data.Dataset):
    """One item: the normalised picture (1 x rows x 501), the standardised engineered numbers, label, weight."""

    def __init__(self, kind, data_dir, names, rows, eng, y, w, augment, zip_path=AUDIO_ZIP):
        self.kind, self.data_dir, self.names, self.rows = kind, Path(data_dir), names, np.asarray(rows)
        self.zip_path, self.augment = str(zip_path), augment
        self.eng = torch.from_numpy(np.asarray(eng, np.float32))
        self.y = torch.from_numpy(np.asarray(y, np.int64))
        self.w = torch.from_numpy(np.asarray(w if w is not None else np.ones(len(rows)), np.float32))
        self.mean = self.std = None
        self._mel = self._zip = None
        rows_ = input_rows(kind)
        self.aug = {**SETTINGS["specaugment"], "freq_width": max(2, SETTINGS["specaugment"]["freq_width"] * rows_ // 128)}

    def __getstate__(self):
        return {**self.__dict__, "_mel": None, "_zip": None}

    def __len__(self):
        return len(self.rows)

    def raw(self, r: int) -> np.ndarray:
        if self.kind == "stft":
            if self._zip is None:
                from shipzip import SplitZip
                self._zip = SplitZip(self.zip_path)
            wave, sr, _ = F.decode_wav(self._zip.read(self.names[r]))
            return stft_db(F.fix_length(F.resample(wave, sr)))
        if self._mel is None:
            self._mel = np.load(self.data_dir / "mel.npy", mmap_mode="r")
        x = np.asarray(self._mel[r], np.float32)
        return dct(x, type=2, norm="ortho", axis=0)[:N_MFCC_IMAGE] if self.kind == "mfcc" else x

    def fit_norm(self, n=600, seed=0):
        """Mean and spread of every row (band, coefficient or bin) over n training clips."""
        pick = np.random.default_rng(seed).choice(len(self.rows), min(n, len(self.rows)), replace=False)
        x = np.stack([self.raw(int(self.rows[i])) for i in pick])
        return x.mean(axis=(0, 2)), np.maximum(x.std(axis=(0, 2)), 1e-3)

    def __getitem__(self, i):
        x = torch.from_numpy(self.raw(int(self.rows[i])))
        x = ((x - self.mean[:, None]) / self.std[:, None]).unsqueeze(0)
        if self.augment:
            x = D.spec_augment(x, self.aug)
        return x, self.eng[i], self.y[i], self.w[i]


class Net(torch.nn.Module):
    """EfficientNet-B0 on the picture; optionally the engineered numbers join before the output layer."""

    def __init__(self, n_eng: int, n_classes: int, pretrained=True, drop_rate=0.3, eng_hidden=128):
        super().__init__()
        import timm
        self.cnn = timm.create_model("efficientnet_b0", pretrained=pretrained, in_chans=1, num_classes=0)
        self.eng = (torch.nn.Sequential(torch.nn.Linear(n_eng, eng_hidden), torch.nn.ReLU(), torch.nn.Dropout(drop_rate))
                    if n_eng else None)
        self.dropout = torch.nn.Dropout(drop_rate)
        self.head = torch.nn.Linear(self.cnn.num_features + (eng_hidden if n_eng else 0), n_classes)

    def forward(self, x, e):
        z = self.dropout(self.cnn(x))
        return self.head(torch.cat([z, self.eng(e)], 1) if self.eng is not None else z)


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    out = []
    for x, e, _, _ in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out.append(torch.softmax(model(x.to(device), e.to(device)).float(), 1).cpu())
    return torch.cat(out).numpy()


def feature_columns(use: bool, eng_cols) -> list[str]:
    if not use:
        return []
    combo = json.loads((W.OUT_DIR / "walk_result.json").read_text())["final"]
    return W.group_cols(W.cost_groups(eng_cols), combo)


def run_name(kind, use_features, fold) -> str:
    return f"f{fold}_{kind}" + ("_features" if use_features else "")


def train(kind="mel", use_features=False, fold=1, data_dir=DATA_DIR, out_dir=OUT_DIR, seed=0, epochs=None, batch_size=0,
          workers=2, limit=0, device="auto", pretrained=True):
    t_start = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    run_dir = Path(out_dir) / run_name(kind, use_features, fold)
    run_dir.mkdir(parents=True, exist_ok=True)
    table, folds, feats, eng_cols = W.load_all(data_dir)
    f = O.GroupFold(table, folds, fold, W.GROUPINGS[W.MAIN])
    rows = {s: f.rows[s] for s in ("train", "validation", "test")}
    y = {s: f.y[s] for s in rows}
    w, ships = f.w["train"], {s: f.ships[s] for s in rows}
    if limit:                                                # quick tests: a few clips of every split
        rng = np.random.default_rng(seed)
        for s in rows:
            keep = np.sort(rng.choice(len(rows[s]), min(limit, len(rows[s])), replace=False))
            rows[s], y[s], ships[s] = rows[s][keep], y[s][keep], ships[s][keep]
            if s == "train":
                w = w[keep]
    cols = feature_columns(use_features, eng_cols)
    X = {s: feats.loc[rows[s], cols].to_numpy(np.float32) if cols else np.zeros((len(rows[s]), 0), np.float32)
         for s in rows}
    if cols:
        mu, sd = X["train"].mean(0), X["train"].std(0) + 1e-6
        X = {s: np.clip((v - mu) / sd, -10, 10) for s, v in X.items()}

    names = table.zip_name.to_numpy()
    ds = {s: Pictures(kind, data_dir, names, rows[s], X[s], y[s], w if s == "train" else None, augment=s == "train")
          for s in rows}
    mean, std = ds["train"].fit_norm(seed=seed)
    for d in ds.values():
        d.mean, d.std = torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(std.astype(np.float32))

    dev = pick_device(device)
    amp = dev.type == "cuda"
    bs = batch_size or max(4, auto_batch_size("efficientnet_b0", dev) * F.N_MELS // max(F.N_MELS, input_rows(kind)))
    kw = dict(num_workers=workers, pin_memory=amp, persistent_workers=workers > 0)
    train_loader = torch.utils.data.DataLoader(ds["train"], bs, shuffle=True,
                                               generator=torch.Generator().manual_seed(seed), **kw)
    loaders = {s: torch.utils.data.DataLoader(ds[s], bs, shuffle=False, **kw) for s in ("validation", "test")}

    model = Net(len(cols), len(NAMES), pretrained, SETTINGS["drop_rate"], SETTINGS["eng_hidden"]).to(dev)
    n_epochs = epochs or SETTINGS["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=SETTINGS["lr"], weight_decay=SETTINGS["weight_decay"])
    steps, warm = n_epochs * len(train_loader), SETTINGS["warmup_epochs"] * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: (k + 1) / warm if k < warm else
                                              0.5 * (1 + math.cos(math.pi * min(1.0, (k - warm) / max(1, steps - warm)))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    guard = TemperatureGuard()
    print(f"{run_dir.name}: {kind} {input_rows(kind)} x {F.N_FRAMES}, {len(cols)} engineered numbers, "
          f"{len(rows['train'])} train / {len(rows['validation'])} val / {len(rows['test'])} test clips, batch {bs}, "
          f"{dev}", flush=True)

    best, best_epoch, patience, log = -1.0, 0, SETTINGS["patience"], []
    for epoch in range(1, n_epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for xb, eb, yb, wb in train_loader:
            xb, eb, yb, wb = xb.to(dev, non_blocking=True), eb.to(dev), yb.to(dev), wb.to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                logits = model(xb, eb)
            loss = (TF.cross_entropy(logits.float(), yb, reduction="none") * wb).sum() / wb.sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            losses.append(loss.item())
            guard.step()
        t_train = time.time() - t0
        val = balanced_accuracy_score(y["validation"], predict(model, loaders["validation"], dev, amp).argmax(1))
        improved = val > best
        if improved:
            best, best_epoch, patience = val, epoch, SETTINGS["patience"]
            save_checkpoint(run_dir / "best.pt", model=model.state_dict(), epoch=epoch, kind=kind, columns=cols,
                            norm={"mean": mean.tolist(), "std": std.tolist()})
        else:
            patience -= 1
        log.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_balanced_accuracy": val,
                    "seconds": round(time.time() - t0, 1), "train_clips_per_s": round(len(rows["train"]) / t_train, 1),
                    "gpu_temp_max": guard.max_seen})
        pd.DataFrame(log).to_csv(run_dir / "log.csv", index=False)
        print(f"epoch {epoch}: train loss {log[-1]['train_loss']:.3f}, validation balanced accuracy {val:.3f}"
              f"{'  *best*' if improved else ''} ({log[-1]['seconds']:.0f} s)", flush=True)
        if patience <= 0:
            break

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=dev, weights_only=True)["model"])
    t0 = time.time()
    probs = predict(model, loaders["test"], dev, amp)
    us = 1e6 * (time.time() - t0) / len(rows["test"])
    res = M.scores(y["test"], probs, ships["test"], NAMES)
    pd.DataFrame(probs, columns=[f"p_{c}" for c in NAMES]).assign(
        row=rows["test"], fold=fold, y=y["test"], ship=ships["test"]).to_parquet(run_dir / "predictions.parquet")
    summary = {"input": kind, "rows": input_rows(kind), "features": use_features, "n_engineered": len(cols),
               "fold": fold, "best_epoch": best_epoch, "val_balanced_accuracy": best, "epochs_run": epoch,
               "batch_size": bs, "settings": SETTINGS, "minutes": round((time.time() - t_start) / 60, 1),
               "predict_us_per_clip_with_loading": us, **res}
    (run_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"test: balanced accuracy {res['balanced_accuracy']:.3f} per clip, "
          f"{res['per_ship_balanced_accuracy']:.3f} per ship (best epoch {best_epoch})", flush=True)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", choices=INPUTS, default="mel")
    p.add_argument("--features", action="store_true", help="add the sets kept by lesson 23's walk")
    p.add_argument("--fold", type=int, default=1, choices=range(1, 6))
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="only N clips per split (quick test)")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = p.parse_args(argv)
    with keep_awake():
        train(a.input, a.features, a.fold, a.data_dir, a.out_dir, epochs=a.epochs, batch_size=a.batch_size,
              workers=a.workers, limit=a.limit)


if __name__ == "__main__":
    main()
