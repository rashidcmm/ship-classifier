"""Unknown-ship detection with the CNN + engineered features (lesson 26), judged per ship.

Lesson 21 (section B) let gradient boosting say "unknown" when it was unsure. Here the same test uses lesson 17's
network: EfficientNet-B0 on the mel spectrogram fused with 139 engineered numbers, five classes (merchant vessels:
cargo + tanker merged). Its five fold models (runs/fusion_cnn/f<k>_16k/best.pt) were trained only on the five known
classes, so the six left-out types (sailing, other, passenger ship, rescue, dredger, pilot vessel) are ships they
have never heard. No new training: the saved models are run on

  - the validation fold (known ships the model was not trained on): sets the threshold,
  - the test fold (known ships, as in lesson 17),
  - the left-out ships of the same fold (open_set.all_folds), so every left-out ship is tested exactly once.

A ship's confidence is its highest class probability after averaging its clips. The threshold keeps 95 % of the
known validation ships; a test ship below it is called "unknown". Per clip is reported too (same rule, clip by clip).

  python src/unknown_cnn.py              # -> runs/unknown_cnn/ (GPU, inference only, about 10-15 minutes)
  python src/unknown_cnn.py --reference  # lesson 21's gradient boosting, scored the same way (CPU, a few minutes)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import five_classes as F5
import fusion_cnn as FN
import open_set as O
import six_classes as S
from device import pick_device
from keepawake import keep_awake
from paths import DATA_DIR, RUNS_DIR

OUT_DIR = RUNS_DIR / "unknown_cnn"
CNN_DIR = FN.OUT_DIR
NAMES = F5.CLASSES5
KEEP_KNOWN = O.KEEP_KNOWN                  # 0.95, as lesson 21
UNKNOWN = "unknown"


def threshold(conf, keep=KEEP_KNOWN) -> float:
    """The confidence that keeps the share `keep` of the known scores at or above it."""
    return float(np.quantile(np.asarray(conf), 1 - keep))


def _per_ship(probs, ships):
    df = pd.DataFrame(probs).assign(ship=np.asarray(ships)).groupby("ship", sort=True).mean()
    p = df.to_numpy()
    return df.index.to_numpy(), p.max(1), p.argmax(1)


def _first(values, ships, index):
    return pd.Series(np.asarray(values)).groupby(np.asarray(ships)).first().reindex(index).to_numpy()


def _auroc(conf_known, conf_unknown) -> float:
    return float(roc_auc_score(np.r_[np.zeros(len(conf_known)), np.ones(len(conf_unknown))],
                               -np.r_[conf_known, conf_unknown]))


def open_set_scores(val_probs, val_ships, known_probs, known_y, known_ships, unknown_probs, unknown_ships,
                    unknown_types, names=NAMES, keep=KEEP_KNOWN) -> dict:
    """Known test ships against never-heard ships; thresholds from the validation (known) ships and clips."""
    names = list(names)
    thr_clip = threshold(val_probs.max(1), keep)
    _, val_conf, _ = _per_ship(val_probs, val_ships)
    thr_ship = threshold(val_conf, keep)

    k_idx, k_conf, k_pred = _per_ship(known_probs, known_ships)
    k_true = _first(known_y, known_ships, k_idx).astype(int)
    u_idx, u_conf, u_pred = _per_ship(unknown_probs, unknown_ships)
    u_type = _first(unknown_types, unknown_ships, u_idx)
    k_acc, u_rej = k_conf >= thr_ship, u_conf < thr_ship

    open_recall = {n: float((k_acc & (k_pred == i))[k_true == i].mean()) for i, n in enumerate(names)}
    open_recall[UNKNOWN] = float(u_rej.mean())
    closed = {n: float((k_pred == i)[k_true == i].mean()) for i, n in enumerate(names)}
    c_known, c_unknown = known_probs.max(1), unknown_probs.max(1)
    return {
        "threshold_clip": thr_clip, "threshold_ship": thr_ship,
        "known_clips_accepted": float((c_known >= thr_clip).mean()),
        "unknown_clips_rejected": float((c_unknown < thr_clip).mean()),
        "auroc_clip": _auroc(c_known, c_unknown), "auroc_ship": _auroc(k_conf, u_conf),
        "known_ships_accepted": float(k_acc.mean()), "unknown_ships_rejected": float(u_rej.mean()),
        "unknown_ships_rejected_by_type": {t: float(u_rej[u_type == t].mean()) for t in sorted(set(u_type))},
        "auroc_ship_by_type": {t: _auroc(k_conf, u_conf[u_type == t]) for t in sorted(set(u_type))},
        "known_ships_rejected_by_class": {n: float((~k_acc)[k_true == i].mean()) for i, n in enumerate(names)},
        "unknown_ships_called_as": {n: float((~u_rej & (u_pred == i)).mean()) for i, n in enumerate(names)},
        "closed_per_ship_recall": closed,
        "closed_per_ship_balanced_accuracy": float(np.mean(list(closed.values()))),
        "open_set_per_ship_recall": open_recall,
        "open_set_per_ship_balanced_accuracy": float(np.mean(list(open_recall.values()))),
        "n_known_ships": int(len(k_idx)), "n_unknown_ships": int(len(u_idx)),
    }


def load_model(run_dir: Path, device):
    ck = torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=True)
    model = FN.FusionNet(len(ck["columns"]), len(NAMES), pretrained=False, drop_rate=FN.SETTINGS["drop_rate"],
                         eng_hidden=FN.SETTINGS["eng_hidden"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


def predict_rows(model, ck, rows, eng, data_dir, device, batch_size=64, workers=0) -> np.ndarray:
    """Class probabilities for table rows, with the checkpoint's own feature and spectrogram normalisation."""
    X = eng.loc[rows, ck["columns"]].to_numpy(np.float32)
    X = np.clip((X - np.asarray(ck["eng_mean"], np.float32)) / np.asarray(ck["eng_std"], np.float32), -10, 10)
    norm = (np.asarray(ck["norm"]["mean"]), np.asarray(ck["norm"]["std"]))
    ds = FN.FusionDataset(data_dir, rows, X, np.zeros(len(rows)), None, norm, ck["n_bands"], augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size, shuffle=False, num_workers=workers,
                                         pin_memory=device.type == "cuda")
    return FN.predict(model, loader, device, amp=device.type == "cuda")


def run(data_dir=DATA_DIR, cnn_dir=CNN_DIR, out_dir=OUT_DIR, folds=range(1, S.N_FOLDS + 1), device="auto",
        batch_size=64, workers=0, top_hz=16000):
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = pick_device(device)
    table = pd.read_parquet(Path(data_dir) / "clips.parquet")
    eng = O.load_engineered(data_dir)
    folds6, folds_all = S.make_folds(table, seed=0), O.all_folds(table)
    labels, ships = table.label.to_numpy(), S.ship_key(table).to_numpy()
    results, parts = [], []
    for k in folds:
        f = F5.Fold5(table, folds6, k)
        model, ck = load_model(Path(cnn_dir) / f"f{k}_{int(top_hz) // 1000}k", dev)
        unknown_rows = np.flatnonzero(np.isin(labels, O.LEFT_OUT) & (folds_all == k))
        rows = {"validation": f.rows["validation"], "test": f.rows["test"], "unknown": unknown_rows}
        probs = {s: predict_rows(model, ck, r, eng, data_dir, dev, batch_size, workers) for s, r in rows.items()}
        res = open_set_scores(probs["validation"], f.ships["validation"], probs["test"], f.y["test"], f.ships["test"],
                              probs["unknown"], ships[unknown_rows], labels[unknown_rows])
        results.append({"fold": k, "n_unknown_clips": int(len(unknown_rows)), **res})
        for s, r in rows.items():
            parts.append(pd.DataFrame(probs[s].astype("float32"), columns=[f"p_{n}" for n in NAMES]).assign(
                row=r, fold=k, split=s, label=labels[r], ship=ships[r]))
        print(f"[{(time.time() - t0) / 60:5.1f} min] fold {k}: AUROC per ship {res['auroc_ship']:.3f} "
              f"(per clip {res['auroc_clip']:.3f}); known ships accepted {res['known_ships_accepted']:.3f}, "
              f"unknown ships rejected {res['unknown_ships_rejected']:.3f}; open-set per ship "
              f"{res['open_set_per_ship_balanced_accuracy']:.3f} (closed {res['closed_per_ship_balanced_accuracy']:.3f})",
              flush=True)
        del model
    pd.concat(parts, ignore_index=True).to_parquet(out_dir / "predictions.parquet")
    summary = {"model": "EfficientNet-B0 + 139 engineered (lesson 17), five classes", "known": NAMES,
               "unknown_types": O.LEFT_OUT, "keep_known": KEEP_KNOWN, "top_hz": top_hz,
               "minutes": round((time.time() - t0) / 60, 1), "results": results}
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    return summary


def reference(data_dir=DATA_DIR, out_dir=OUT_DIR, folds=range(1, S.N_FOLDS + 1), seed=0):
    """Lesson 21's gradient boosting (section B: 144 engineered numbers, five classes, fitted on the training folds,
    the validation fold sets the threshold), scored with the same per-ship rules as the CNN. CPU."""
    from threadpoolctl import threadpool_limits
    t0 = time.time()
    table = pd.read_parquet(Path(data_dir) / "clips.parquet")
    eng = O.load_engineered(data_dir)
    cols, folds_all = list(eng.columns), O.all_folds(table, seed)
    labels, ships = table.label.to_numpy(), S.ship_key(table).to_numpy()
    results = []
    with threadpool_limits(4):
        for k in folds:
            f = O.GroupFold(table, folds_all, k, O.FIVE)
            model, _, _ = S.fit_gb(f.X(eng, cols, "train"), f.y["train"], f.w["train"], f.X(eng, cols, "validation"),
                                   f.y["validation"], f.w["validation"], seed)
            unknown_rows = np.flatnonzero(np.isin(labels, O.LEFT_OUT) & (folds_all == k))
            res = open_set_scores(model.predict_proba(f.X(eng, cols, "validation")), f.ships["validation"],
                                  model.predict_proba(f.X(eng, cols, "test")), f.y["test"], f.ships["test"],
                                  model.predict_proba(eng.loc[unknown_rows, cols].to_numpy()), ships[unknown_rows],
                                  labels[unknown_rows], names=list(O.FIVE))
            results.append({"fold": k, **res})
            print(f"[{(time.time() - t0) / 60:5.1f} min] reference fold {k}: AUROC per ship {res['auroc_ship']:.3f}, "
                  f"open-set per ship {res['open_set_per_ship_balanced_accuracy']:.3f}", flush=True)
    summary = {"model": "gradient boosting, 144 engineered (lesson 21 B)", "known": list(O.FIVE),
               "keep_known": KEEP_KNOWN, "results": results}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "reference.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--cnn-dir", type=Path, default=CNN_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--reference", action="store_true", help="lesson 21's gradient boosting instead, scored per ship (CPU)")
    a = p.parse_args(argv)
    with keep_awake():
        if a.reference:
            reference(a.data_dir, a.out_dir)
        else:
            run(a.data_dir, a.cnn_dir, a.out_dir, device=a.device, batch_size=a.batch_size, workers=a.workers)


if __name__ == "__main__":
    main()
