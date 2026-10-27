"""One-minute speed test on this machine -> estimated training hours for every experiment in the plan.

  python src/benchmark.py                       # default backbone, auto batch size
  python src/benchmark.py --backbone mobilenetv3_large_100
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D
from device import auto_batch_size, pick_device
from keepawake import keep_awake
from labels import CLASSES, known_classes, load_config
from models import BACKBONES, ShipNet
from paths import DATA_DIR

EPOCH_FRACTION = 0.8          # early stopping usually ends runs before the maximum epochs

# experiment -> (number of training runs, task, select_rows options)
EXPERIMENTS = {
    "E2": (1, "class", {}),
    "E3": (3, "class", {"holdout": "A"}),
    "ensemble": (2, "class", {"holdout": "A"}),     # seeds 1 and 2 of group A: ensembles vs softmax for unknowns
    "E5": (1, "class", {"split_mode": "leaky"}),
    "E7": (1, "class", {"hydrophone_shift": True}),
    "E9": (1, "distance", {}),
    "E10": (1, "speed", {}),
}


def measure(model, loader, steps, device, train):
    amp = device.type == "cuda"
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4) if train else None
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    model.train(train)
    it, n, t0 = iter(loader), 0, None
    for step in range(steps + 2):                 # 2 warm-up batches are not timed
        if step == 2:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0, n = time.time(), 0
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        x, y = b["x"].to(device), b["y_class"].to(device)
        with torch.set_grad_enabled(train), torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            loss = F.cross_entropy(model(x)["logits"].float(), y)
        if train:
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        n += len(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return n / (time.time() - t0)


def main(argv=None) -> dict:
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", choices=BACKBONES, default=cfg["backbone"])
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=cfg["workers"])
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    a = p.parse_args(argv)

    with keep_awake():
        try:
            mel, table = D.open_features(a.data_dir)
        except D.DataError as e:
            sys.exit(f"error: {e}")
        device = pick_device(a.device)
        bs = a.batch_size or auto_batch_size(a.backbone, device)
        rows = D.select_rows(table, fold=1, split="train", task="class")
        norm = D.compute_norm(mel, rows, max_clips=200)
        ds = D.MelDataset(a.data_dir, table, rows, CLASSES, norm, augment=True, aug_cfg=cfg["specaugment"])
        loader = torch.utils.data.DataLoader(ds, bs, shuffle=True, num_workers=a.workers,
                                             persistent_workers=a.workers > 0)
        model = ShipNet(a.backbone, len(CLASSES), pretrained=not a.no_pretrained).to(device)
        print(f"benchmarking {a.backbone} on {device}, batch {bs}, {a.steps} batches each for training and evaluation ...")
        train_cps = measure(model, loader, a.steps, device, train=True)
        eval_cps = measure(model, loader, a.steps, device, train=False)

    res = {"device": str(device), "backbone": a.backbone, "batch_size": bs,
           "train_clips_per_s": round(train_cps, 1), "eval_clips_per_s": round(eval_cps, 1)}
    print(f"\ntraining: {train_cps:.0f} clips/s | evaluation: {eval_cps:.0f} clips/s\n")
    print(f"{'experiment':<10} {'runs':>4} {'train clips':>11} {'min/epoch':>9} {'hours':>7}")
    total = 0.0
    for name, (runs, task, extra) in EXPERIMENTS.items():
        sel = dict(fold=1, task=task, **extra)
        n_tr = len(D.select_rows(table, split="train", **sel))
        n_va = len(D.select_rows(table, split="validation", **sel))
        epoch_min = (n_tr / train_cps + n_va / eval_cps) / 60
        hours = runs * cfg["epochs"][task] * EPOCH_FRACTION * epoch_min / 60
        res[name] = hours
        total += hours
        print(f"{name:<10} {runs:>4} {n_tr:>11} {epoch_min:>9.1f} {hours:>7.1f}")
    res["total_hours"] = total
    print(f"{'total':<10} {'':>4} {'':>11} {'':>9} {total:>7.1f}\n"
          f"(estimates assume {EPOCH_FRACTION:.0%} of the maximum epochs because of early stopping)")
    return res


if __name__ == "__main__":
    main()
