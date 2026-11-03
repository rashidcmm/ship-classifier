"""EfficientNet-B0 on the mel spectrogram, fused with the engineered features (lesson 17).

Five classes (cargo + tanker merged, lesson 16), lesson 13's folds, test ships and ship weights. One network:

    mel spectrogram (1 x bands x 501) -> EfficientNet-B0 (ImageNet start) -> 1280 numbers -> dropout --+
                                                                                                         +-> 5 classes
    139 engineered numbers (lesson 15's best set: all groups but loudness and impulsiveness) -> 128 ----+

Two versions: all 128 mel bands (10 Hz - 16 kHz), or only the bands that lie entirely below 8 kHz (101 bands).
Trained on the training folds, the epoch kept is the one with the best balanced accuracy on the validation fold, and
the test fold is scored per clip and per ship (lesson 16). For reference, gradient boosting is trained on the same
139 numbers with lesson 13's protocol (--reference, CPU).

  python src/fusion_cnn.py --top-hz 16000          # -> runs/fusion_cnn/f1_16k/ (GPU, about 30 minutes)
  python src/fusion_cnn.py --top-hz 8000           # -> runs/fusion_cnn/f1_8k/
  python src/fusion_cnn.py --reference             # -> runs/fusion_cnn/reference_f1.json (CPU, a few minutes)
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
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D
import five_classes as F5
import six_classes as S
import six_classes_fewer as B
from device import TemperatureGuard, auto_batch_size, pick_device
from keepawake import keep_awake
from models import save_checkpoint
from paths import DATA_DIR, RUNS_DIR

OUT_DIR = RUNS_DIR / "fusion_cnn"
DROP_GROUP = "loudness and impulsiveness"           # lesson 15: the only group whose removal did not cost
SETTINGS = dict(epochs=12, patience=3, lr=3e-4, weight_decay=0.01, warmup_epochs=1, drop_rate=0.3, eng_hidden=128,
                specaugment={"freq_masks": 2, "freq_width": 16, "time_masks": 2, "time_width": 50})


def engineered_columns(columns) -> list[str]:
    return [c for g, cols in S.feature_groups(columns).items() if g != DROP_GROUP for c in cols]


class FusionNet(torch.nn.Module):
    def __init__(self, n_eng: int, n_classes: int, pretrained: bool = True, drop_rate: float = 0.3,
                 eng_hidden: int = 128):
        super().__init__()
        import timm
        self.cnn = timm.create_model("efficientnet_b0", pretrained=pretrained, in_chans=1, num_classes=0)
        self.eng = torch.nn.Sequential(torch.nn.Linear(n_eng, eng_hidden), torch.nn.ReLU(),
                                       torch.nn.Dropout(drop_rate))
        self.dropout = torch.nn.Dropout(drop_rate)
        self.head = torch.nn.Linear(self.cnn.num_features + eng_hidden, n_classes)

    def forward(self, x, e):
        return self.head(torch.cat([self.dropout(self.cnn(x)), self.eng(e)], dim=1))


class FusionDataset(torch.utils.data.Dataset):
    """One item: normalised spectrogram (first n_bands bands), standardised engineered numbers, label, weight."""

    def __init__(self, data_dir, rows, eng, y, w, norm, n_bands, augment):
        self.mel_path, self.rows, self.n_bands, self.augment = Path(data_dir) / "mel.npy", np.asarray(rows), n_bands, augment
        self.eng = torch.from_numpy(np.asarray(eng, np.float32))
        self.y = torch.from_numpy(np.asarray(y, np.int64))
        self.w = torch.from_numpy(np.asarray(w if w is not None else np.ones(len(rows)), np.float32))
        self.mean = torch.from_numpy(norm[0][:n_bands, None].astype(np.float32))
        self.std = torch.from_numpy(norm[1][:n_bands, None].astype(np.float32))
        self._mel = None

    def __getstate__(self):
        return {**self.__dict__, "_mel": None}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self._mel is None:
            self._mel = np.load(self.mel_path, mmap_mode="r")
        x = torch.from_numpy(np.asarray(self._mel[self.rows[i], :self.n_bands], np.float32))
        x = ((x - self.mean) / self.std).unsqueeze(0)
        if self.augment:
            x = D.spec_augment(x, SETTINGS["specaugment"])
        return x, self.eng[i], self.y[i], self.w[i]


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    out = []
    for x, e, _, _ in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out.append(torch.softmax(model(x.to(device), e.to(device)).float(), 1).cpu())
    return torch.cat(out).numpy()


def train_fusion(top_hz=16000, fold=1, data_dir=DATA_DIR, out_dir=OUT_DIR, seed=0, epochs=None, batch_size=0,
                 workers=2, limit=0, device="auto", pretrained=True):
    t_start = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    run_dir = Path(out_dir) / f"f{fold}_{int(top_hz) // 1000}k"
    run_dir.mkdir(parents=True, exist_ok=True)
    table, _, eng = S.load_features(data_dir)
    f = F5.Fold5(table, S.make_folds(table, seed=0), fold)
    if limit:                                             # quick tests: a few clips of every split
        rng = np.random.default_rng(seed)
        for s in ("train", "validation", "test"):
            keep = np.sort(rng.choice(len(f.rows[s]), min(limit, len(f.rows[s])), replace=False))
            f.rows[s], f.y[s], f.y6[s], f.ships[s], f.w[s] = (f.rows[s][keep], f.y[s][keep], f.y6[s][keep],
                                                              f.ships[s][keep], f.w[s][keep])
    cols = engineered_columns(eng.columns)
    X = {s: eng.loc[f.rows[s], cols].to_numpy(np.float32) for s in ("train", "validation", "test")}
    mu, sd = X["train"].mean(0), X["train"].std(0) + 1e-6
    X = {s: np.clip((v - mu) / sd, -10, 10) for s, v in X.items()}
    n_bands = B.bands_below(top_hz)
    mel, _ = D.open_features(data_dir)
    norm = D.compute_norm(mel, f.rows["train"], seed=seed)

    dev = pick_device(device)
    amp = dev.type == "cuda"
    bs = batch_size or auto_batch_size("efficientnet_b0", dev)
    kw = dict(num_workers=workers, pin_memory=amp, persistent_workers=workers > 0)
    ds = {s: FusionDataset(data_dir, f.rows[s], X[s], f.y[s], f.w["train"] if s == "train" else None, norm,
                           n_bands, augment=s == "train") for s in ("train", "validation", "test")}
    train_loader = torch.utils.data.DataLoader(ds["train"], bs, shuffle=True,
                                               generator=torch.Generator().manual_seed(seed), **kw)
    loaders = {s: torch.utils.data.DataLoader(ds[s], bs, shuffle=False, **kw) for s in ("validation", "test")}

    model = FusionNet(len(cols), len(F5.CLASSES5), pretrained, SETTINGS["drop_rate"], SETTINGS["eng_hidden"]).to(dev)
    n_epochs = epochs or SETTINGS["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=SETTINGS["lr"], weight_decay=SETTINGS["weight_decay"])
    steps, warm = n_epochs * len(train_loader), SETTINGS["warmup_epochs"] * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: (k + 1) / warm if k < warm else
                                              0.5 * (1 + math.cos(math.pi * min(1.0, (k - warm) / max(1, steps - warm)))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    guard = TemperatureGuard()
    print(f"{run_dir.name}: {n_bands} mel bands, {len(cols)} engineered numbers, {len(f.rows['train'])} train / "
          f"{len(f.rows['validation'])} val / {len(f.rows['test'])} test clips, batch {bs}, {dev}", flush=True)

    best, best_epoch, patience, log = -1.0, 0, SETTINGS["patience"], []
    for epoch in range(1, n_epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for x, e, y, w in train_loader:
            x, e, y, w = x.to(dev, non_blocking=True), e.to(dev), y.to(dev), w.to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                logits = model(x, e)
            loss = (F.cross_entropy(logits.float(), y, reduction="none") * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            losses.append(loss.item())
            guard.step()
        val = balanced_accuracy_score(f.y["validation"], predict(model, loaders["validation"], dev, amp).argmax(1))
        improved = val > best
        if improved:
            best, best_epoch, patience = val, epoch, SETTINGS["patience"]
            save_checkpoint(run_dir / "best.pt", model=model.state_dict(), epoch=epoch, n_bands=n_bands,
                            columns=cols, eng_mean=mu.tolist(), eng_std=sd.tolist(),
                            norm={"mean": norm[0].tolist(), "std": norm[1].tolist()})
        else:
            patience -= 1
        log.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_balanced_accuracy": val,
                    "seconds": round(time.time() - t0, 1), "gpu_temp_max": guard.max_seen})
        pd.DataFrame(log).to_csv(run_dir / "log.csv", index=False)
        print(f"epoch {epoch}: train loss {log[-1]['train_loss']:.3f}, validation balanced accuracy {val:.3f}"
              f"{'  *best*' if improved else ''} ({log[-1]['seconds']:.0f} s)", flush=True)
        if patience <= 0:
            break

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=dev, weights_only=True)["model"])
    probs = predict(model, loaders["test"], dev, amp)
    res = F5.scores5(f.y["test"], probs, f.ships["test"])
    pd.DataFrame(probs, columns=[f"p_{c}" for c in F5.CLASSES5]).assign(
        row=f.rows["test"], fold=fold, y=f.y["test"], y6=f.y6["test"], ship=f.ships["test"]).to_parquet(
        run_dir / "predictions.parquet")
    summary = {"model": "EfficientNet-B0 + engineered", "top_hz": top_hz, "n_bands": n_bands, "fold": fold,
               "n_engineered": len(cols), "best_epoch": best_epoch, "val_balanced_accuracy": best,
               "epochs_run": epoch, "batch_size": bs, "settings": SETTINGS,
               "minutes": round((time.time() - t_start) / 60, 1), **res}
    (run_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"test: balanced accuracy {res['balanced_accuracy']:.3f} per clip, "
          f"{res['per_ship_balanced_accuracy']:.3f} per ship (best epoch {best_epoch})", flush=True)
    return summary


def reference(fold=1, data_dir=DATA_DIR, out_dir=OUT_DIR, seed=0):
    """Gradient boosting and logistic regression on the same 139 numbers, lesson 13's protocol, five classes."""
    from threadpoolctl import threadpool_limits
    table, _, eng = S.load_features(data_dir)
    f = F5.Fold5(table, S.make_folds(table, seed=0), fold)
    cols = engineered_columns(eng.columns)
    out = []
    with threadpool_limits(4):
        for model in S.MODELS:
            probs, setting, val, _ = S.fit_and_test(f, eng, cols, model, seed)
            out.append({"model": model, "fold": fold, "n_features": len(cols), "setting": setting,
                        "val_balanced_accuracy": val, **F5.scores5(f.y["test"], probs, f.ships["test"])})
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"reference_f{fold}.json").write_text(json.dumps(out, indent=2))
    for r in out:
        print(f"{r['model']}: {r['balanced_accuracy']:.3f} per clip, {r['per_ship_balanced_accuracy']:.3f} per ship")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top-hz", type=int, default=16000, choices=B.TOP_HZ)
    p.add_argument("--fold", type=int, default=1, choices=range(1, 6))
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="only N clips per split (quick test)")
    p.add_argument("--reference", action="store_true", help="gradient boosting / LR on the same numbers instead")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = p.parse_args(argv)
    with keep_awake():
        if a.reference:
            reference(a.fold, a.data_dir, a.out_dir)
        else:
            train_fusion(a.top_hz, a.fold, a.data_dir, a.out_dir, epochs=a.epochs, batch_size=a.batch_size,
                         workers=a.workers, limit=a.limit)


if __name__ == "__main__":
    main()
