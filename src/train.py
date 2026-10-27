"""Train one model. Every experiment in the plan is one call of this script (lesson 05 explains each part).

Examples
  python src/train.py --task class                         # E2: 12-class model on fold 1
  python src/train.py --task class --holdout A             # E3: fishing/sailing/tanker held out as "unknown"
  python src/train.py --task class --merge-into-other pilotvessel dredger rescue   # 9 classes
  python src/train.py --task distance --init-from runs/class_f1_efficientnet_b0_vessel/best.pt
  python src/train.py --task class --resume                # continue after Ctrl+C, a crash or --max-hours

Output: runs/<name>/ with config.json, norm.npz, log.csv, last.pt (every epoch), best.pt, done.json.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D
from device import TemperatureGuard, auto_batch_size, pick_device
from keepawake import keep_awake
from labels import MERGEABLE, known_classes, load_config
from metrics import macro_f1
from models import BACKBONES, PretrainedDownloadError, ShipNet, save_checkpoint
from paths import DATA_DIR, RUNS_DIR

HUBER_DELTA = 0.2
LOG_FIELDS = ["epoch", "train_loss", "val_loss", "val_metric", "lr", "seconds", "clips_per_s",
              "gpu_temp_max", "paused_s"]


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=["class", "distance", "speed", "multi"])
    p.add_argument("--fold", type=int, default=1, choices=range(1, 6))
    p.add_argument("--backbone", choices=BACKBONES, help="default: configs/default.yaml")
    p.add_argument("--holdout", choices=["A", "B", "C"], help="hold out an open-set group (unknown ships)")
    p.add_argument("--split-mode", default="vessel", choices=["vessel", "leaky"])
    p.add_argument("--merge-into-other", nargs="+", metavar="CLASS", choices=MERGEABLE,
                   help="train and test these classes as 'other' (e.g. those with few ships: pilotvessel dredger rescue)")
    p.add_argument("--hydrophone-shift", action="store_true", help="train on 2523, test on 2556")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, help="default: per task in configs/default.yaml")
    p.add_argument("--batch-size", type=int, default=0, help="0 = choose from GPU memory")
    p.add_argument("--lr", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--workers", type=int)
    p.add_argument("--init-from", type=Path, help="start from another run's best.pt (e.g. the class model)")
    p.add_argument("--no-pretrained", action="store_true", help="random start instead of ImageNet weights")
    p.add_argument("--resume", action="store_true", help="continue this run from its last.pt")
    p.add_argument("--max-hours", type=float, help="stop cleanly after this many hours (resume later)")
    p.add_argument("--max-gpu-temp", type=int)
    p.add_argument("--limit-train", type=int, default=0, help="use only N training clips (quick tests)")
    p.add_argument("--limit-val", type=int, default=0)
    p.add_argument("--config", type=Path, help="YAML file whose settings replace those in configs/default.yaml")
    p.add_argument("--name", help="run folder name (default: built from the options)")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    return p


def run_name(args, backbone) -> str:
    name = f"{args.task}_f{args.fold}_{backbone}_{args.split_mode}"
    if args.holdout:
        name += f"_h{args.holdout}"
    if args.hydrophone_shift:
        name += "_hyd"
    if args.merge_into_other:
        name += "_merged"
    if args.seed:
        name += f"_s{args.seed}"
    return name


def load_init_weights(model, path: Path) -> list[str]:
    """Copy every weight whose name and shape match; return the names that were skipped."""
    src = torch.load(path, map_location="cpu", weights_only=True)["model"]
    own = model.state_dict()
    skipped = [k for k in src if k not in own or own[k].shape != src[k].shape]
    model.load_state_dict({k: v for k, v in src.items() if k not in skipped}, strict=False)
    return skipped


def code_version() -> str:
    h = hashlib.sha256()
    for f in sorted(Path(__file__).resolve().parent.glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def subsample(rows, n, seed):
    if n and len(rows) > n:
        return np.sort(np.random.default_rng(seed).choice(rows, n, replace=False))
    return rows


def masked_huber(pred, target):
    ok = torch.isfinite(target)
    if not ok.any():
        return pred.sum() * 0.0
    return F.huber_loss(pred[ok].float(), target[ok], delta=HUBER_DELTA)


def compute_loss(task, out, batch, class_weights):
    y = batch["y_class"]
    if task in ("class", "multi"):
        ce = F.cross_entropy(out["logits"].float(), y, weight=class_weights, ignore_index=-1)
        if task == "class":
            return ce
        return ce + masked_huber(out["distance"], batch["y_dist"]) + masked_huber(out["speed"], batch["y_speed"])
    return masked_huber(out[task], batch["y_dist" if task == "distance" else "y_speed"])


def lr_lambda_factory(warmup_steps, total_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return f


@torch.no_grad()
def validate(model, loader, task, class_weights, device, amp):
    model.eval()
    losses, preds, ys, err = [], [], [], []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(b["x"])
        losses.append(compute_loss(task, out, b, class_weights).item())
        if task in ("class", "multi"):
            preds.append(out["logits"].argmax(1).cpu()); ys.append(b["y_class"].cpu())
        else:
            t = b["y_dist" if task == "distance" else "y_speed"]
            ok = torch.isfinite(t)
            err.append((out[task].float()[ok] - t[ok]).abs().cpu())
    if task in ("class", "multi"):
        y, p = torch.cat(ys).numpy(), torch.cat(preds).numpy()
        metric = macro_f1(y, p, n_classes=len(class_weights))
    else:
        scale = 1000.0 if task == "distance" else 10.0           # back to metres / knots
        metric = float(torch.cat(err).mean()) * scale
    return float(np.mean(losses)), metric


def rng_state(gen):
    state = {"torch": torch.get_rng_state(), "python": random.getstate(), "loader": gen.get_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state, gen):
    torch.set_rng_state(state["torch"].cpu())            # RNG states must be CPU tensors
    random.setstate(state["python"])
    gen.set_state(state["loader"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def append_log(path: Path, rows: list[dict]) -> list[dict]:
    """Append rows to log.csv; returns the rows that could not be written (file locked, e.g. open in Excel)."""
    new_file = not path.exists()
    try:
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, LOG_FIELDS)
            if new_file:
                w.writeheader()
            w.writerows(rows)
        return []
    except PermissionError:
        print(f"warning: cannot write {path.name} (is it open in Excel?) - training continues, will retry next epoch")
        return rows


SAME_SETUP = ("fold", "split_mode", "hydrophone_shift")


def check_same_setup(source_cfg: dict, args) -> None:
    """--init-from must come from a run with the same fold and split, or its weights have seen our test clips."""
    diff = [k for k in SAME_SETUP if source_cfg.get(k, getattr(args, k)) != getattr(args, k)]
    if diff:
        sys.exit(f"--init-from {args.init_from}: that run used a different {', '.join(diff)} "
                 f"({', '.join(f'{k}={source_cfg.get(k)}' for k in diff)}); its weights were trained on clips that are "
                 f"in this run's test set. Use the class run of the same fold/split.")


def run(args) -> Path:
    cfg = load_config()
    if args.config:
        cfg.update(load_config(args.config))
    backbone = args.backbone or cfg["backbone"]
    epochs = args.epochs if args.epochs is not None else cfg["epochs"][args.task]
    lr = args.lr or cfg["lr"]
    weight_decay = args.weight_decay if args.weight_decay is not None else cfg["weight_decay"]
    workers = args.workers if args.workers is not None else cfg["workers"]
    max_temp = args.max_gpu_temp or cfg["max_gpu_temp"]
    run_dir = Path(args.runs_dir) / (args.name or run_name(args, backbone))

    if args.resume and not (run_dir / "last.pt").exists():
        sys.exit(f"--resume: no last.pt in {run_dir}; start without --resume.")
    if args.resume and (run_dir / "done.json").exists():
        sys.exit(f"{run_dir.name} already finished (see done.json); nothing to resume. "
                 f"To train again, choose a new --name.")
    if args.init_from and not args.resume:
        check_same_setup(torch.load(args.init_from, map_location="cpu", weights_only=True).get("config", {}), args)
    if not args.resume and (run_dir / "log.csv").exists():
        sys.exit(f"Run folder {run_dir} already exists. Use --resume to continue it, or --name to start a new one.")
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    mel, table = D.open_features(args.data_dir)
    if args.merge_into_other:
        table = D.merge_into_other(table, args.merge_into_other)
    known = [c for c in known_classes(args.holdout) if c not in (args.merge_into_other or [])]
    sel = dict(fold=args.fold, task=args.task, holdout=args.holdout, split_mode=args.split_mode,
               hydrophone_shift=args.hydrophone_shift, seed=args.seed)
    train_rows = subsample(D.select_rows(table, split="train", **sel), args.limit_train, args.seed)
    val_rows = subsample(D.select_rows(table, split="validation", **sel), args.limit_val, args.seed)

    if args.resume:
        z = np.load(run_dir / "norm.npz")
        norm = (z["mean"], z["std"])
    else:
        norm = D.compute_norm(mel, train_rows, max_clips=cfg["norm_clips"], seed=args.seed)
        np.savez(run_dir / "norm.npz", mean=norm[0], std=norm[1])

    device = pick_device(args.device)
    amp = device.type == "cuda"
    batch_size = args.batch_size or auto_batch_size(backbone, device)
    gen = torch.Generator().manual_seed(args.seed)
    loader_kw = dict(num_workers=workers, pin_memory=amp, persistent_workers=workers > 0)
    train_ds = D.MelDataset(args.data_dir, table, train_rows, known, norm, augment=True, aug_cfg=cfg["specaugment"])
    val_ds = D.MelDataset(args.data_dir, table, val_rows, known, norm, augment=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True, generator=gen, **loader_kw)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size, shuffle=False, **loader_kw)

    counts = np.bincount(train_ds.y_class[train_rows][train_ds.y_class[train_rows] >= 0], minlength=len(known))
    weights = np.where(counts > 0, counts.sum() / (len(known) * np.maximum(counts, 1)), 0.0)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    model = ShipNet(backbone, len(known), pretrained=not (args.no_pretrained or args.resume),
                    drop_rate=cfg["drop_rate"]).to(device)
    if args.init_from and not args.resume:
        skipped = load_init_weights(model, args.init_from)
        print(f"initialised from {args.init_from} (skipped: {', '.join(skipped) or 'none'})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(cfg["warmup_epochs"] * steps_per_epoch, epochs * steps_per_epoch))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    higher_better = args.task in ("class", "multi")
    start_epoch, best, patience_left = 1, -math.inf if higher_better else math.inf, cfg["patience"]

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(backbone=backbone, epochs=epochs, lr=lr, weight_decay=weight_decay, batch_size=batch_size,
                  drop_rate=cfg["drop_rate"], specaugment=cfg["specaugment"], device=str(device), known=known, n_train=int(len(train_rows)),
                  n_val=int(len(val_rows)), code_version=code_version(), run_dir=str(run_dir))
    if args.resume:
        # load on CPU: load_state_dict moves weights/optimizer state to the device itself, and RNG states must stay on CPU
        ckpt = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt["scaler"]:                                   # empty when the run started on a CPU
            scaler.load_state_dict(ckpt["scaler"])
        set_rng_state(ckpt["rng"], gen)
        start_epoch, best, patience_left = ckpt["epoch"] + 1, ckpt["best_metric"], ckpt["patience_left"]
        print(f"resuming {run_dir.name} at epoch {start_epoch} (best so far {best:.4f})")
    else:
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "last.pt.tmp").unlink(missing_ok=True)       # leftover from a crash mid-save

    norm_state = {"mean": norm[0].tolist(), "std": norm[1].tolist()}
    guard = TemperatureGuard(max_c=max_temp, resume_c=cfg["resume_gpu_temp"])
    session_start = time.time()
    print(f"{run_dir.name}: {len(train_rows)} train / {len(val_rows)} val clips, {len(known)} classes, "
          f"batch {batch_size}, device {device}, epochs {start_epoch}..{epochs}")

    stop_reason, pending_log = None, []
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t0, paused, losses = time.time(), 0.0, []
        for b in train_loader:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(b["x"])
                loss = compute_loss(args.task, out, b, class_weights)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
            paused += guard.step()
        train_secs = time.time() - t0
        val_loss, metric = validate(model, val_loader, args.task, class_weights, device, amp)

        improved = metric > best if higher_better else metric < best
        if improved:
            best, patience_left = metric, cfg["patience"]
            save_checkpoint(run_dir / "best.pt", model=model.state_dict(), config=config, known=known,
                            norm=norm_state, epoch=epoch, best_metric=best)
        else:
            patience_left -= 1
        save_checkpoint(run_dir / "last.pt", model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), config=config,
                        known=known, norm=norm_state, epoch=epoch, best_metric=best,
                        patience_left=patience_left, rng=rng_state(gen))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "val_metric": metric,
               "lr": scheduler.get_last_lr()[0], "seconds": round(time.time() - t0, 1),
               "clips_per_s": round(len(train_rows) / max(train_secs - paused, 1e-6), 1),
               "gpu_temp_max": guard.max_seen, "paused_s": round(paused, 1)}
        pending_log = append_log(run_dir / "log.csv", pending_log + [row])
        unit = "macro-F1" if higher_better else ("MAE m" if args.task == "distance" else "MAE knots")
        print(f"epoch {epoch}/{epochs}  train loss {row['train_loss']:.4f}  val loss {val_loss:.4f}  "
              f"val {unit} {metric:.4f}{'  *best*' if improved else ''}  ({row['seconds']} s)")

        if patience_left <= 0:
            stop_reason = f"early stop: no improvement for {cfg['patience']} epochs"
            break
        if args.max_hours is not None and (time.time() - session_start) / 3600 >= args.max_hours and epoch < epochs:
            if append_log(run_dir / "log.csv", pending_log):
                print("log rows could not be written; close log.csv and check the printed epoch lines above")
            print(f"stopped for time after epoch {epoch}; continue with the same command plus --resume")
            return run_dir

    if append_log(run_dir / "log.csv", pending_log):
        print("log rows could not be written; close log.csv and check the printed epoch lines above")
    (run_dir / "done.json").write_text(json.dumps(
        {"reason": stop_reason or "all epochs done", "best_metric": best, "epochs_run": epoch}, indent=2))
    print(f"finished: best val {best:.4f} -> {run_dir}")
    return run_dir


def main(argv=None):
    with keep_awake():
        try:
            run(make_parser().parse_args(argv))
        except (D.DataError, PretrainedDownloadError, FileNotFoundError) as e:
            sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
