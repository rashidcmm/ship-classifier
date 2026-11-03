"""Fewer features for the six classes (lesson 15): fewer and lower mel bands, and whole feature groups.

Two questions, both asked with lesson 13's setup (same folds, ship weights, logistic regression and gradient
boosting, settings chosen on the validation fold, scores on the never-touched test fold):

  1. bands: do the models need the high frequencies, and all 128 mel bands? The mel spectrogram of lesson 02
     (128 bands, 10 Hz - 16 kHz) is cut at a top frequency (16, 8, 4, 2 or 1 kHz: only the bands that lie entirely
     below it are kept), and neighbouring bands are merged (1, 2, 4 or 8 at a time: 128 -> 64, 32, 16 bands). Every
     version gives two feature sets: the mean and spread of each band (lesson 13's simple set) and MFCCs computed
     from the kept bands only (mean, spread and how fast each changes, coefficients 1-19).
  2. groups: lesson 13's 144 engineered features come in 7 groups (MFCC, 1/3-octave levels, ...). Whole groups are
     swapped, never single features: each group is left out once, and groups are added one at a time, always the
     one that helps most on the validation folds, to find the smallest set that does as well as all seven.

  python src/six_classes_fewer.py features   # -> data/features6_bands.parquet (a few minutes; needs data/mel.npy)
  python src/six_classes_fewer.py bands      # -> runs/six_classes_fewer/bands_results.json
  python src/six_classes_fewer.py groups     # -> runs/six_classes_fewer/groups_results.json, groups_forward.csv
  python src/six_classes_fewer.py all        # the three steps in a row
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import dct
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D
import features as F
import six_classes as S
from keepawake import keep_awake
from paths import DATA_DIR, RUNS_DIR

OUT_DIR = RUNS_DIR / "six_classes_fewer"
BANDS_FILE = "features6_bands.parquet"
TOP_HZ = (16000, 8000, 4000, 2000, 1000)
MERGE = (1, 2, 4, 8)
N_MFCC = 19
TOLERANCE = 0.005          # "as good as": within half a point of the best mean validation score (as in lesson 13)
# On the i5-12500H laptop, the default (one thread per logical core, on mixed fast and slow cores) made logistic
# regression 35 times slower (110 s against 3 s for one fit) and gradient boosting about twice as slow; 8 threads
# are no faster than 4.
THREADS = 4

# band b of the mel spectrogram is a triangle from MEL_EDGES[b] to MEL_EDGES[b + 2], peaking at MEL_EDGES[b + 1]
MEL_EDGES = F.mel_to_hz(np.linspace(F.hz_to_mel(F.FMIN), F.hz_to_mel(F.FMAX), F.N_MELS + 2))


# ---------------------------------------------------------------- band features

def bands_below(top_hz: float) -> int:
    """How many of the 128 mel bands lie entirely below top_hz (they are the lowest ones)."""
    return int((MEL_EDGES[2:] <= top_hz * (1 + 1e-9)).sum())


def n_groups(top_hz: float, merge: int) -> int:
    """Merged bands (merge neighbours each, counted from the lowest band) that lie entirely below top_hz."""
    return bands_below(top_hz) // merge


def group_edges(merge: int) -> list[tuple[float, float]]:
    """Frequency range (Hz) covered by each merged band."""
    return [(float(MEL_EDGES[g * merge]), float(MEL_EDGES[(g + 1) * merge + 1])) for g in range(F.N_MELS // merge)]


def mfcc_tag(top_hz: float, merge: int) -> str:
    return f"mfcc{int(top_hz) // 1000}k_{merge}"


def variant_columns(columns, top_hz: float, merge: int, kind: str) -> list[str]:
    """The columns of one version: kind 'bands' (mean and spread of each kept band) or 'mfcc'."""
    if kind == "bands":
        return [f"band{merge}_{g}_{s}" for g in range(n_groups(top_hz, merge)) for s in ("mean", "std")]
    prefix = mfcc_tag(top_hz, merge) + "_"
    return [c for c in columns if c.startswith(prefix)]


def _delta_std(c: np.ndarray) -> np.ndarray:
    """Spread of the frame-to-frame change of each coefficient; c is (clips, coefficients, frames)."""
    pad = np.pad(c, ((0, 0), (0, 0), (2, 2)), mode="edge")
    return ((pad[..., 3:-1] - pad[..., 1:-3] + 2 * (pad[..., 4:] - pad[..., :-4])) / 10).std(axis=2)


def band_features(mel, rows, chunk: int = 250) -> pd.DataFrame:
    """Every band and MFCC version for the given rows of the mel spectrogram (dB, (clips, 128, frames))."""
    rows = np.asarray(rows)
    parts = []
    for s in range(0, len(rows), chunk):
        x = np.asarray(mel[rows[s:s + chunk]], dtype=np.float32)
        power = 10.0 ** (x / 10.0)
        out = {}
        for m in MERGE:
            n = F.N_MELS // m
            db = x if m == 1 else 10 * np.log10(
                power[:, :n * m].reshape(len(x), n, m, -1).sum(axis=2) + F.POWER_FLOOR)
            mean, std = db.mean(axis=2), db.std(axis=2)
            for g in range(n):
                out[f"band{m}_{g}_mean"], out[f"band{m}_{g}_std"] = mean[:, g], std[:, g]
            for top in TOP_HZ:
                kept = n_groups(top, m)
                c = dct(db[:, :kept], type=2, norm="ortho", axis=1)[:, 1:min(N_MFCC, kept - 1) + 1]
                c_mean, c_std, c_change = c.mean(axis=2), c.std(axis=2), _delta_std(c)
                tag = mfcc_tag(top, m)
                for j in range(c.shape[1]):
                    out[f"{tag}_{j + 1}_mean"], out[f"{tag}_{j + 1}_std"] = c_mean[:, j], c_std[:, j]
                    out[f"{tag}_{j + 1}_change"] = c_change[:, j]
        parts.append(pd.DataFrame(out).astype("float32"))
    df = pd.concat(parts, ignore_index=True)
    df.insert(0, "row", rows)
    return df


def build_band_features(data_dir=DATA_DIR) -> Path:
    t0 = time.time()
    mel, table = D.open_features(data_dir)
    rows = np.flatnonzero(table.label.isin(S.CLASSES6).to_numpy())
    df = band_features(mel, rows)
    path = Path(data_dir) / BANDS_FILE
    df.to_parquet(path)
    print(f"band features: {df.shape[1] - 1} columns for {len(df):,} clips, {time.time() - t0:.0f} s", flush=True)
    return path


# ---------------------------------------------------------------- shared

def _folds(table, seed):
    folds = S.make_folds(table, seed=seed)
    return [S.Fold(table, folds, k) for k in range(1, S.N_FOLDS + 1)]


def _test(f, feats, cols, model, seed, **info) -> dict:
    probs, setting, val, _ = S.fit_and_test(f, feats, cols, model, seed)
    return {**info, "model": model, "fold": f.k, "n_features": len(cols), "setting": setting,
            "val_balanced_accuracy": val, **S.scores(f.y["test"], probs, f.ships["test"])}


def _save(path: Path, results: list, **extra):
    path.write_text(json.dumps({"classes": S.CLASSES6, **extra, "results": results}, indent=2))


def _logger(t0):
    def log(msg):
        print(f"[{(time.time() - t0) / 60:5.1f} min] {msg}", flush=True)
    return log


def smallest_as_good(val: pd.Series, size: pd.Series, tolerance: float = TOLERANCE):
    """Index of the smallest entry whose (mean validation) score is within tolerance of the best."""
    ok = val[val >= val.max() - tolerance].index
    return min(ok, key=lambda i: (size[i], -val[i]))


# ---------------------------------------------------------------- 1. fewer and lower mel bands

def band_versions(tops=TOP_HZ, merges=MERGE) -> list[tuple[int, int]]:
    """(top Hz, merge) pairs; the two plain sweeps first (cut the top; merge bands), then the rest of the grid."""
    first = [(t, merges[0]) for t in tops] + [(tops[0], m) for m in merges[1:]]
    return first + [(t, m) for t in tops for m in merges if (t, m) not in first]


def run_bands(data_dir=DATA_DIR, out_dir=OUT_DIR, seed=0, tops=TOP_HZ, merges=MERGE, models=S.MODELS):
    t0, log = time.time(), _logger(time.time())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bands_results.json"
    table = pd.read_parquet(Path(data_dir) / "clips.parquet")
    feats = pd.read_parquet(Path(data_dir) / BANDS_FILE).set_index("row")
    fold_list = _folds(table, seed)
    results = json.loads(path.read_text())["results"] if path.exists() else []    # continue an interrupted run
    done = {(r["kind"], r["top_hz"], r["merge"], r["model"]) for r in results}
    for top, m in band_versions(tops, merges):
        for kind in ("bands", "mfcc"):
            cols = variant_columns(feats.columns, top, m, kind)
            for model in models:
                if (kind, top, m, model) in done:
                    continue
                for f in fold_list:
                    results.append(_test(f, feats, cols, model, seed, kind=kind, top_hz=top, merge=m,
                                         n_bands=n_groups(top, m)))
                _save(path, results, seed=seed)
                ba = np.mean([r["balanced_accuracy"] for r in results[-len(fold_list):]])
                log(f"{kind:5s} up to {top / 1000:4.0f} kHz, {n_groups(top, m):3d} bands, {model}: {ba:.3f}")
    summary = {"seed": seed, "minutes": round((time.time() - t0) / 60, 1),
               "edges_hz": {str(m): group_edges(m) for m in merges}}
    _save(path, results, **summary)
    log("bands finished")
    return {**summary, "results": results}


# ---------------------------------------------------------------- 2. whole feature groups

def run_groups(data_dir=DATA_DIR, out_dir=OUT_DIR, seed=0, models=S.MODELS):
    t0, log = time.time(), _logger(time.time())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table, _, eng = S.load_features(data_dir)
    groups = S.feature_groups(eng.columns)                      # all 19 MFCCs, as chosen in lesson 13
    names = list(groups)
    fold_list = _folds(table, seed)

    def cols_of(chosen):
        return [c for g in names if g in chosen for c in groups[g]]

    results, forward = [], []
    for model in models:
        # leave one group out (and all seven, for reference)
        for out in [None, *names]:
            kept = [g for g in names if g != out]
            for f in fold_list:
                results.append(_test(f, eng, cols_of(kept), model, seed, kind="leave one out",
                                     left_out=out or "(none)", groups=kept, n_groups=len(kept)))
            log(f"{model}: without {out or '(none)'}")
        # add one group at a time: the one with the best mean validation score
        chosen = []
        while len(chosen) < len(names):
            remaining = [g for g in names if g not in chosen]
            val = {g: float(np.mean([S.validation_score(f, eng, cols_of(chosen + [g]), model, seed)
                                     for f in fold_list])) for g in remaining} if len(remaining) > 1 else {}
            best = max(val, key=val.get) if val else remaining[0]
            chosen.append(best)
            forward += [{"model": model, "step": len(chosen), "candidate": g, "val_balanced_accuracy": v,
                         "chosen": g == best} for g, v in val.items()]
            info = dict(kind="forward", step=len(chosen), added=best, groups=list(chosen), n_groups=len(chosen))
            if len(chosen) == len(names):                       # all seven: same columns as above, same result
                results += [{**r, **info} for r in results
                            if r["kind"] == "leave one out" and r["model"] == model and r["left_out"] == "(none)"]
            else:
                for f in fold_list:
                    results.append(_test(f, eng, cols_of(chosen), model, seed, **info))
            _save(out_dir / "groups_results.json", results, seed=seed)
            pd.DataFrame(forward).to_csv(out_dir / "groups_forward.csv", index=False)
            log(f"{model}: step {len(chosen)} adds {best}")
    summary = {"seed": seed, "groups": {g: len(c) for g, c in groups.items()},
               "minutes": round((time.time() - t0) / 60, 1)}
    _save(out_dir / "groups_results.json", results, **summary)
    log("groups finished")
    return {**summary, "results": results}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["features", "bands", "groups", "all"])
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--threads", type=int, default=THREADS, help="maths-library threads (see THREADS)")
    a = p.parse_args(argv)
    with keep_awake(), threadpool_limits(a.threads):
        if a.step in ("features", "all") and not (a.step == "all" and (a.data_dir / BANDS_FILE).exists()):
            build_band_features(a.data_dir)
        if a.step in ("bands", "all"):
            df = pd.DataFrame(run_bands(a.data_dir, a.out_dir)["results"])
            print(df.groupby(["kind", "top_hz", "merge", "model"]).balanced_accuracy.mean().unstack().round(3))
        if a.step in ("groups", "all"):
            df = pd.DataFrame(run_groups(a.data_dir, a.out_dir)["results"])
            print(df.groupby(["kind", "model", "n_groups"]).balanced_accuracy.mean().round(3))


if __name__ == "__main__":
    main()
