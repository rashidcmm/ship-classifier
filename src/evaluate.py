"""Score a finished run on its test split and draw the report figures (lessons 07-10).

  python src/evaluate.py runs/class_f1_efficientnet_b0_vessel
  python src/evaluate.py runs/class_f1_efficientnet_b0_vessel_hA --mc 20 --ensemble runs/..._s1 runs/..._s2
  python src/evaluate.py runs/class_f1_efficientnet_b0_vessel --noise 20 10 0     # needs the audio zip

Writes <run>/eval/: metrics.json, predictions.parquet, calibration.json (temperature + "unknown" thresholds,
chosen on the VALIDATION split) and PNG figures.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D
import metrics as M
import reliability as R
from device import auto_batch_size, pick_device
from keepawake import keep_awake
from labels import load_config
from models import load_model
from paths import AUDIO_ZIP, DATA_DIR

DIST_BANDS = [700, 1400]          # metres: near / mid / far
SPEED_BANDS = [6, 12]             # knots: slow / medium / fast
KEEP_KNOWN = 95                   # thresholds accept 95 % of known validation clips


# ---------- prediction ----------
@torch.no_grad()
def predict_rows(model, data_dir, table, rows, known, norm, device, batch_size, workers=0) -> dict:
    ds = D.MelDataset(data_dir, table, rows, known, norm, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size, shuffle=False, num_workers=workers)
    model.to(device).eval()
    out = {"logits": [], "distance": [], "speed": []}
    for b in loader:
        o = model(b["x"].to(device))
        for k in out:
            out[k].append(o[k].float().cpu())
    res = {k: torch.cat(v).numpy() for k, v in out.items()}
    res["loader"] = loader
    return res


def calibrate(val_logits, y_val) -> dict:
    """Temperature and 'unknown' thresholds from validation clips, with every class weighted equally
    (fold 1's validation split is 38 % dredger, which would otherwise dominate)."""
    w = M.class_balanced_weights(y_val)
    T = R.fit_temperature(val_logits, y_val, weights=w)
    return {"temperature": T,
            "msp_threshold": M.weighted_percentile(R.msp_score(val_logits, T), w, KEEP_KNOWN),
            "energy_threshold": M.weighted_percentile(R.energy_score(val_logits, T), w, KEEP_KNOWN),
            "keep_known_percent": KEEP_KNOWN, "chosen_on": "validation (class-balanced)"}


def add_noise(wave, snr_db, rng) -> np.ndarray:
    """White Gaussian noise at a given signal-to-noise ratio (relative to the clip's own power)."""
    power = np.mean(wave.astype(np.float64) ** 2)
    noise = rng.normal(0.0, np.sqrt(power / 10 ** (snr_db / 10)), size=wave.shape)
    return (wave + noise).astype(np.float32)


# ---------- figures ----------
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(cm, names, path):
    with np.errstate(invalid="ignore"):
        norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(np.nan_to_num(norm), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right"); ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Confusion matrix (test, counts; colour = row %)")
    _save(fig, path)


def plot_reliability(probs_before, probs_after, y, n_bins, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, probs, title in zip(axes, (probs_before, probs_after), ("before", "after temperature scaling")):
        b = M.reliability_bins(probs, y, n_bins)
        centers = (np.arange(n_bins) + 0.5) / n_bins
        ax.bar(centers, np.nan_to_num(b["acc"]), width=1 / n_bins, edgecolor="k", alpha=0.7, label="accuracy")
        ax.plot([0, 1], [0, 1], "r--", label="perfect calibration")
        ax.set_title(f"{title}  (ECE {M.ece(probs, y, n_bins):.3f})"); ax.set_xlabel("confidence")
    axes[0].set_ylabel("accuracy"); axes[0].legend()
    _save(fig, path)


def plot_roc(scores, is_unknown, path):
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(is_unknown, s)
        ax.plot(fpr, tpr, label=f"{name} (AUROC {M.open_set_auroc(s, is_unknown):.2f})")
    ax.plot([0, 1], [0, 1], "k:"); ax.set_xlabel("known flagged as unknown"); ax.set_ylabel("unknown caught")
    ax.set_title("Unknown-ship detection (test)"); ax.legend()
    _save(fig, path)


def plot_scatter(true, pred, unit, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true, pred, s=4, alpha=0.4)
    lim = [0, max(np.max(true), np.max(pred)) * 1.05]
    ax.plot(lim, lim, "r--"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"true ({unit})"); ax.set_ylabel(f"predicted ({unit})")
    _save(fig, path)


# ---------- main evaluation ----------
def evaluate_run(run_dir, *, ensemble=(), mc_samples=0, noise_snrs=(), audio_zip=None, noise_limit=1000,
                 split="test", data_dir=None, device="auto", workers=0) -> dict:
    run_dir = Path(run_dir)
    if noise_snrs and not Path(audio_zip or AUDIO_ZIP).exists():
        raise FileNotFoundError(f"--noise needs the audio zip ({Path(audio_zip or AUDIO_ZIP)}); copy "
                                f"gdrive_download/audio/ to this machine, or pass --audio-zip / set SHIPSOUND_AUDIO_ZIP")
    out_dir = run_dir / "eval"
    out_dir.mkdir(exist_ok=True)
    best = run_dir / "best.pt"
    fingerprint = f"{best.stat().st_size}-{best.stat().st_mtime_ns}"     # changes when the model is retrained
    model, info = load_model(run_dir)
    cfg, known, norm = info["config"], info["known"], info["norm"]
    data_dir = Path(data_dir or (cfg["data_dir"] if Path(cfg["data_dir"]).exists() else DATA_DIR))
    mel, table = D.open_features(data_dir)
    dev = pick_device(device)
    bs = auto_batch_size(cfg["backbone"], dev) * 2
    task = cfg["task"]
    sel = dict(fold=cfg["fold"], task=task, holdout=cfg["holdout"], split_mode=cfg["split_mode"],
               hydrophone_shift=cfg["hydrophone_shift"], seed=cfg["seed"])
    test_rows = D.select_rows(table, split=split, **sel)
    test = predict_rows(model, data_dir, table, test_rows, known, norm, dev, bs, workers)
    sub = table.iloc[test_rows].reset_index(drop=True)
    index = {c: i for i, c in enumerate(known)}
    y = np.array([index.get(c, -1) for c in sub.label])
    pred = pd.DataFrame({"row": test_rows, "label": sub.label, "y_class": y})
    result = {"run": run_dir.name, "task": task, "split": split, "n_eval": int(len(test_rows)),
              "model_fingerprint": fingerprint}

    if task in ("class", "multi"):
        val_rows = D.select_rows(table, split="validation", **sel)
        val = predict_rows(model, data_dir, table, val_rows, known, norm, dev, bs, workers)
        y_val = np.array([index[c] for c in table.label.iloc[val_rows]])
        calibration = calibrate(val["logits"], y_val)
        T = calibration["temperature"]
        (out_dir / "calibration.json").write_text(json.dumps(calibration, indent=2))

        logits = test["logits"]
        p1, pT = R.softmax(logits), R.softmax(logits, T)
        for i, c in enumerate(known):
            pred[f"logit_{c}"] = logits[:, i]
        pred["pred_class"] = [known[i] for i in logits.argmax(1)]
        pred["msp"], pred["energy"] = R.msp_score(logits, T), R.energy_score(logits, T)
        k = y >= 0
        yk, pk = y[k], logits[k].argmax(1)
        ships = sub[k & (sub.label != "background").to_numpy()]
        n_ships = ships.groupby("label").MMSI.nunique()
        result["closed_set"] = {
            "accuracy": M.accuracy(yk, pk), "macro_f1": M.macro_f1(yk, pk, len(known)),
            "per_class_recall": {c: (None if np.isnan(r) else float(r))
                                 for c, r in zip(known, M.per_class_recall(yk, pk, len(known)))},
            "n_test": int(k.sum()),
            "rare_class_warning": sorted(c for c in known if c != "background" and n_ships.get(c, 0) < 5)}
        plot_confusion(M.confusion(yk, pk, len(known)), known, out_dir / "confusion.png")
        bins = load_config()["ece_bins"]
        result["calibration"] = {
            "temperature": T,
            "ece_before": M.ece(p1[k], yk, bins), "ece_after": M.ece(pT[k], yk, bins),
            "nll_before": M.nll(p1[k], yk), "nll_after": M.nll(pT[k], yk),
            "brier_before": M.brier(p1[k], yk), "brier_after": M.brier(pT[k], yk)}
        plot_reliability(p1[k], pT[k], yk, bins, out_dir / "reliability_before_after.png")

        dist = sub.mean_distance.to_numpy(float)
        band = np.digitize(dist, DIST_BANDS)
        result["by_distance"] = {}
        for b, name in enumerate(("near <700 m", "mid 700-1400 m", "far >1400 m")):
            m_ = k & np.isfinite(dist) & (band == b)
            result["by_distance"][name] = {"n": int(m_.sum()),
                                          "accuracy": M.accuracy(y[m_], logits[m_].argmax(1)) if m_.any() else None}
        plot_by_distance(result["by_distance"], out_dir / "accuracy_by_distance.png")

        scores = {"msp": pred["msp"].to_numpy(), "energy": pred["energy"].to_numpy()}
        if mc_samples:
            s = R.mc_dropout_logits(model, test["loader"], mc_samples, dev)
            ps = R.softmax(s)
            pred["mc_entropy"], pred["mc_mi"] = R.entropy(ps.mean(0)), R.mutual_information(ps)
            scores.update(mc_entropy=pred["mc_entropy"].to_numpy(), mc_mi=pred["mc_mi"].to_numpy())
            pm = ps.mean(0)
            result["mc_dropout"] = {"samples": mc_samples, "accuracy": M.accuracy(yk, pm[k].argmax(1)),
                                    "ece": M.ece(pm[k], yk, bins)}
        if ensemble:
            members = [p1]
            for other in ensemble:
                m2, info2 = load_model(other)
                setup = ("fold", "split_mode", "holdout", "hydrophone_shift")
                if info2["known"] != known or any(info2["config"].get(k) != cfg.get(k) for k in setup):
                    raise ValueError(f"ensemble member {Path(other).name} was trained on a different setup "
                                     f"(classes, fold, split mode, holdout or hydrophone shift) than {run_dir.name}")
                members.append(R.softmax(predict_rows(m2, data_dir, table, test_rows, known, info2["norm"],
                                                      dev, bs, workers)["logits"]))
            pe = R.ensemble_probs(members)
            pred["ens_entropy"] = R.entropy(pe)
            scores["ens_entropy"] = pred["ens_entropy"].to_numpy()
            result["ensemble"] = {"members": [run_dir.name] + [Path(e).name for e in ensemble],
                                  "accuracy": M.accuracy(yk, pe[k].argmax(1)), "ece": M.ece(pe[k], yk, bins),
                                  "macro_f1": M.macro_f1(yk, pe[k].argmax(1), len(known))}

        unknown = y == -1
        if unknown.any():
            os_ = {name: {"auroc": M.open_set_auroc(s, unknown), "fpr95": M.fpr_at_95_tpr(s, unknown)}
                   for name, s in scores.items()}
            pc = logits.argmax(1)
            os_["open_set_accuracy_msp"] = M.open_set_accuracy(pc, scores["msp"], calibration["msp_threshold"], y)
            os_["open_set_accuracy_energy"] = M.open_set_accuracy(pc, scores["energy"],
                                                                  calibration["energy_threshold"], y)
            os_["n_unknown"] = int(unknown.sum())
            result["open_set"] = os_
            plot_roc(scores, unknown, out_dir / "open_set_roc.png")

        if noise_snrs:
            result["noise"] = noise_test(model, sub, k, y, known, norm, T, calibration["energy_threshold"],
                                         noise_snrs, Path(audio_zip or AUDIO_ZIP), noise_limit, dev, out_dir)

    pred["pred_distance_m"] = test["distance"] * 1000.0
    pred["pred_speed_kn"] = test["speed"] * 10.0
    regression = {}
    for target, col, scale, bands, unit in (("distance", "mean_distance", 1000.0, DIST_BANDS, "m"),
                                            ("speed", "mean_sog", 10.0, SPEED_BANDS, "knots")):
        if task not in (target, "multi"):
            continue
        true = sub[col].to_numpy(float)
        ok = np.isfinite(true) & (sub.label != "background").to_numpy()
        guess = test[target][ok] * scale
        train_rows = D.select_rows(table, split="train", **{**sel, "task": target})
        baseline = float(np.median(table[col].to_numpy(float)[train_rows]))
        rep = M.regression_report(true[ok], guess, bands)
        rep.update(unit=unit, n=int(ok.sum()), baseline_median=baseline,
                   baseline_mae=float(np.mean(np.abs(true[ok] - baseline))))
        if target == "distance":            # also scored as distance classes (250 m bins, near / far)
            rep.update(M.distance_bin_report(true[ok], guess, near=DIST_BANDS[0]))
            rep["baseline_bins"] = M.distance_bin_report(true[ok], np.full(int(ok.sum()), baseline), near=DIST_BANDS[0])
        regression[target] = rep
        plot_scatter(true[ok], guess, unit, out_dir / f"{target}_scatter.png")
    if regression:
        result["regression"] = regression

    merge_previous(out_dir, result, pred)
    pred.to_parquet(out_dir / "predictions.parquet", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def merge_previous(out_dir, result, pred):
    """Keep sections/columns from earlier evaluations of the SAME model and split (e.g. --mc/--ensemble results
    when evaluating again with --noise). Results of an older, retrained model are dropped."""
    old_path = out_dir / "metrics.json"
    if not old_path.exists():
        return
    old = json.loads(old_path.read_text())
    if old.get("model_fingerprint") != result["model_fingerprint"] or old.get("split") != result["split"]:
        return
    for key, value in old.items():
        result.setdefault(key, value)
    pred_path = out_dir / "predictions.parquet"
    if pred_path.exists():
        old_pred = pd.read_parquet(pred_path)
        if len(old_pred) == len(pred) and (old_pred.row.to_numpy() == pred.row.to_numpy()).all():
            for col in old_pred.columns.difference(pred.columns):
                pred[col] = old_pred[col].to_numpy()


def plot_by_distance(by_distance, path):
    names = list(by_distance)
    acc = [by_distance[n]["accuracy"] or 0 for n in names]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(names, acc, color="tab:blue")
    for i, n in enumerate(names):
        ax.text(i, acc[i] + 0.01, f"n={by_distance[n]['n']}", ha="center", fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("accuracy"); ax.set_title("Accuracy by ship distance (test)")
    _save(fig, path)


def noise_test(model, sub, known_mask, y, known, norm, T, energy_threshold, snrs, audio_zip, limit, dev, out_dir):
    """Re-compute features from the raw audio with added white noise and see how accuracy and 'unknown' flags change."""
    import features as F
    from shipzip import SplitZip
    if not audio_zip.exists():
        raise FileNotFoundError(f"--noise needs the audio zip ({audio_zip}); pass --audio-zip or set SHIPSOUND_AUDIO_ZIP")
    z = SplitZip(str(audio_zip))
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(known_mask)
    idx = np.sort(rng.choice(idx, min(limit, len(idx)), replace=False))
    waves = [F.fix_length(F.resample(*F.decode_wav(z.read(sub.zip_name.iat[i]))[:2])) for i in idx]
    logmel = F.LogMel().to(dev)
    mean = torch.from_numpy(norm[0])[:, None].to(dev); std = torch.from_numpy(norm[1])[:, None].to(dev)
    res = {}
    for snr in [None, *snrs]:
        logits = []
        for s in range(0, len(waves), 32):
            w = np.stack([add_noise(x, snr, rng) if snr is not None else x for x in waves[s:s + 32]])
            m = (logmel(torch.from_numpy(w).to(dev)) - mean) / std
            with torch.no_grad():
                logits.append(model(m.unsqueeze(1))["logits"].float().cpu())
        lg = torch.cat(logits).numpy()
        res["clean" if snr is None else f"{snr} dB"] = {
            "accuracy": M.accuracy(y[idx], lg.argmax(1)),
            "flagged_unknown": float(np.mean(R.energy_score(lg, T) > energy_threshold)), "n": int(len(idx))}
    fig, ax = plt.subplots(figsize=(5, 3.5))
    names = list(res)
    ax.plot(names, [res[n]["accuracy"] for n in names], "o-", label="accuracy")
    ax.plot(names, [res[n]["flagged_unknown"] for n in names], "s--", label="flagged unknown")
    ax.set_ylim(0, 1); ax.set_xlabel("added noise"); ax.legend(); ax.set_title("Robustness to noise")
    _save(fig, out_dir / "noise_curve.png")
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--ensemble", type=Path, nargs="*", default=[], help="other runs (same classes) to average with")
    p.add_argument("--mc", type=int, default=0, help="Monte Carlo dropout samples (e.g. 20)")
    p.add_argument("--noise", type=float, nargs="*", default=[], help="SNRs in dB, e.g. 20 10 0 (needs audio zip)")
    p.add_argument("--audio-zip", type=Path)
    p.add_argument("--noise-limit", type=int, default=1000)
    p.add_argument("--split", default="test", choices=["validation", "test"])
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args(argv)
    if not (a.run_dir / "best.pt").exists():
        sys.exit(f"error: {a.run_dir} has no best.pt - is the run name right, and has training finished an epoch?")
    with keep_awake():
        try:
            res = evaluate_run(a.run_dir, ensemble=a.ensemble, mc_samples=a.mc, noise_snrs=tuple(a.noise),
                               audio_zip=a.audio_zip, noise_limit=a.noise_limit, split=a.split,
                               data_dir=a.data_dir, device=a.device, workers=a.workers)
        except (D.DataError, FileNotFoundError, ValueError) as e:
            sys.exit(f"error: {e}")
    print(json.dumps({k: v for k, v in res.items() if k != "closed_set"} | (
        {"closed_set": {k: v for k, v in res["closed_set"].items() if k != "per_class_recall"}}
        if "closed_set" in res else {}), indent=2))
    print(f"\nfull results and figures in {Path(a.run_dir) / 'eval'}")


if __name__ == "__main__":
    main()
