"""Six ship classes, interpretable sound features and two classical models (lesson 13).

Decided on 1 November 2026: keep background, cargo, fishing, pleasure craft, tanker and tug. Passenger ship is dropped
(110 ships; every class kept has at least 233). Every class has the same 5 h of audio, but very different numbers of
ships (cargo 2,522 ships with 1-2 clips each, tanker 233 with about 15), so:
  - the folds are made again, so that every test fold holds about a fifth of every class's ships (grouped by ship,
    background by recording day), and every ship is tested exactly once;
  - training weights make every class count the same, and every ship the same within its class. No clip is deleted.
Two feature sets go into the same two models (logistic regression and gradient boosting):
  - simple: mean and spread of each mel band, plus overall loudness (257 numbers, lesson 03), no MFCC;
  - engineered: interpretable numbers computed from the raw audio (engineered_features below).

  python src/six_classes.py features      # -> data/features6_simple.parquet, data/features6_engineered.parquet
  python src/six_classes.py run           # -> runs/six_classes/ (5 folds, CPU)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.ndimage import median_filter
from scipy.signal import butter, detrend, find_peaks, sosfilt, welch
from scipy.stats import kurtosis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D
from baseline_classical import band_stats, features_for
from features import decode_wav, fix_length, mel_filterbank, resample
from keepawake import keep_awake
from paths import AUDIO_ZIP, DATA_DIR, RUNS_DIR

CLASSES6 = ["background", "cargo", "fishing", "pleasurecraft", "tanker", "tug"]
N_FOLDS = 5
OUT_DIR = RUNS_DIR / "six_classes"
SIMPLE_FILE, ENGINEERED_FILE = "features6_simple.parquet", "features6_engineered.parquet"


# ---------------------------------------------------------------- folds and weights

def ship_key(table: pd.DataFrame) -> pd.Series:
    """Who a clip belongs to: the ship's MMSI, or the recording day for background (no ship)."""
    mmsi = table.MMSI.map(lambda m: f"mmsi{m:.0f}")
    return mmsi.where(table.label != "background", "day" + table.date.astype(str))


def make_folds(table: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = 0) -> np.ndarray:
    """Fold 1..n_folds for every clip of the six classes (0 for the other classes). Every ship (or background day)
    lands in exactly one fold, and every fold gets about a fifth of every class."""
    keep = np.flatnonzero(table.label.isin(CLASSES6).to_numpy())
    sub = table.iloc[keep]
    folds = np.zeros(len(table), dtype=int)
    splitter = StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
    for k, (_, test) in enumerate(splitter.split(sub, sub.label, ship_key(sub)), start=1):
        folds[keep[test]] = k
    return folds


def split_rows(folds: np.ndarray, k: int, n_folds: int = N_FOLDS) -> dict[str, np.ndarray]:
    """Fold k is the test set, the next fold the validation set (for choosing settings), the other three train."""
    val = k % n_folds + 1
    return {"train": np.flatnonzero((folds > 0) & (folds != k) & (folds != val)),
            "validation": np.flatnonzero(folds == val), "test": np.flatnonzero(folds == k)}


def ship_weights(labels, ships) -> np.ndarray:
    """Training weight per clip: every class gets the same total, and inside a class every ship the same share,
    split over its clips (a tanker with 15 clips gives each clip 1/15 of a ship). Scaled to an average of 1."""
    df = pd.DataFrame({"label": np.asarray(labels), "ship": np.asarray(ships)})
    clips_of_ship = df.groupby("ship")["ship"].transform("size")
    ships_in_class = df.groupby("label")["ship"].transform("nunique")
    w = 1.0 / (clips_of_ship * ships_in_class)
    return (w / w.mean()).to_numpy()


# ---------------------------------------------------------------- engineered features

SR = 32_000
N_FFT, HOP = 2048, 512                 # 64 ms windows (15.6 Hz apart), moved 16 ms each step
HANN = np.hanning(N_FFT)
FREQS = np.fft.rfftfreq(N_FFT, 1 / SR)
LOW = np.searchsorted(FREQS, 20.0)     # ignore the bins below 20 Hz (hydrophone and flow rumble)
WELCH_SEG = 16_384                     # 0.5 s segments for the long-term spectrum: bins 2 Hz apart
N_MEL, N_MFCC = 40, 19                 # MFCCs 1..19 are stored; the models keep the first few (lesson 13, section 5)
MEL_FB = mel_filterbank(SR, N_FFT, N_MEL, 20.0, SR / 2)
THIRD_OCTAVE_NAMES = ["20", "25", "31.5", "40", "50", "63", "80", "100", "125", "160", "200", "250", "315", "400",
                      "500", "630", "800", "1000", "1250", "1600", "2000", "2500", "3150", "4000", "5000", "6300",
                      "8000", "10000", "12500"]
THIRD_OCTAVES = 1000.0 * 2.0 ** (np.arange(-17, 12) / 3)     # exact centres of the bands named above
OCTAVES = 1000.0 * 2.0 ** np.arange(-5, 4)                   # 31 Hz ... 8 kHz
TONAL_RANGE, TONAL_MIN_DB = (10.0, 1000.0), 6.0             # machinery lines: peaks 6 dB above the local spectrum
DEMON_BAND, DEMON_RATE = (2000.0, 10000.0), 400             # propeller noise band; envelope sampled at 400 Hz
DEMON_SOS = butter(4, DEMON_BAND, btype="bandpass", fs=SR, output="sos")
DEMON_EDGES = np.geomspace(1.0, 150.0, 9)                   # 8 bands of modulation rate, 1-150 Hz

GROUPS = {"mfcc": "MFCC", "third": "1/3-octave levels", "octstd": "level changes over time",
          "shape": "spectral shape", "level": "loudness and impulsiveness", "tonal": "tonal lines",
          "demon": "propeller modulation (DEMON)"}


def _stats(name, v):
    return {f"{name}_mean": v.mean(), f"{name}_std": v.std(),
            f"{name}_p10": np.percentile(v, 10), f"{name}_p90": np.percentile(v, 90)}


def _band_db(power, freqs, lo, hi, axis=-1):
    sel = (freqs >= lo) & (freqs < hi)
    if not sel.any():                                        # band narrower than one bin: take the nearest bin
        sel = np.zeros_like(freqs, dtype=bool)
        sel[np.argmin(np.abs(freqs - np.sqrt(lo * hi)))] = True
    return 10 * np.log10(np.take(power, np.flatnonzero(sel), axis=axis).sum(axis=axis) + 1e-20)


def engineered_features(wave: np.ndarray) -> dict[str, float]:
    """One 5-s clip at 32 kHz -> named numbers, in seven groups (prefix before the first '_', see GROUPS)."""
    x = wave.astype(np.float64) - wave.mean()
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP]
    P = np.abs(np.fft.rfft(frames * HANN, axis=1)) ** 2       # power spectrogram, (309 frames, 1025 bins)
    out = {}

    # MFCC: log-mel energies (40 bands, 20 Hz - 16 kHz) -> DCT; mean, spread and how fast each one changes
    logmel = 10 * np.log10(P @ MEL_FB.T + 1e-20)
    c = dct(logmel, type=2, norm="ortho", axis=1)[:, 1:N_MFCC + 1]
    pad = np.pad(c, ((2, 2), (0, 0)), mode="edge")
    delta = (pad[3:-1] - pad[1:-3] + 2 * (pad[4:] - pad[:-4])) / 10
    for k in range(N_MFCC):
        out[f"mfcc_{k + 1}_mean"], out[f"mfcc_{k + 1}_std"] = c[:, k].mean(), c[:, k].std()
        out[f"mfcc_{k + 1}_change"] = delta[:, k].std()

    # long-term spectrum (Welch, 2 Hz bins): level in each 1/3-octave band
    fw, psd = welch(x, SR, window="hann", nperseg=WELCH_SEG, noverlap=WELCH_SEG // 2)
    df = fw[1] - fw[0]
    for name, fc in zip(THIRD_OCTAVE_NAMES, THIRD_OCTAVES):
        out[f"third_{name}Hz"] = _band_db(psd * df, fw, fc * 2 ** (-1 / 6), fc * 2 ** (1 / 6))

    # how much the level of each octave band changes during the clip (steady hum or fluctuating)
    for fc in OCTAVES:
        out[f"octstd_{fc:.0f}Hz"] = _band_db(P, FREQS, fc / np.sqrt(2), fc * np.sqrt(2), axis=1).std()

    # spectral shape, frame by frame: where the energy sits, how spread, how noise-like, how fast it changes
    S, f = P[:, LOW:], FREQS[LOW:]
    p = S / (S.sum(1, keepdims=True) + 1e-30)
    centroid = p @ f
    out |= _stats("shape_centroid", centroid)
    out |= _stats("shape_bandwidth", np.sqrt(np.maximum(p @ f ** 2 - centroid ** 2, 0)))
    out |= _stats("shape_rolloff", f[np.argmax(np.cumsum(p, axis=1) >= 0.85, axis=1)])
    out |= _stats("shape_flatness", np.exp(np.log(S + 1e-30).mean(1)) / (S.mean(1) + 1e-30))
    out |= _stats("shape_flux", np.sqrt((np.diff(p, axis=0) ** 2).sum(1)))
    out |= _stats("shape_energy", 10 * np.log10(S.sum(1) + 1e-20))
    out |= _stats("shape_zcr", (np.diff(np.sign(frames), axis=1) != 0).mean(1))

    # loudness, spectral slope and impulsiveness
    density = np.array([out[f"third_{n}Hz"] for n in THIRD_OCTAVE_NAMES]) - 10 * np.log10(
        THIRD_OCTAVES * (2 ** (1 / 6) - 2 ** (-1 / 6)))      # band level -> level per Hz
    fit = (THIRD_OCTAVES >= 50) & (THIRD_OCTAVES <= 10_000)
    out["level_broadband"] = _band_db(psd * df, fw, 20.0, SR / 2)
    out["level_slope_db_per_decade"] = np.polyfit(np.log10(THIRD_OCTAVES[fit]), density[fit], 1)[0]
    out["level_low_minus_high"] = (density[(THIRD_OCTAVES >= 20) & (THIRD_OCTAVES <= 200)].mean()
                                   - density[(THIRD_OCTAVES >= 2000) & (THIRD_OCTAVES <= 10_000)].mean())
    out["level_kurtosis"] = kurtosis(x)
    out["level_crest_db"] = 20 * np.log10(np.abs(x).max() / (x.std() + 1e-20) + 1e-20)

    # tonal lines (LOFAR): narrow peaks standing out of the long-term spectrum below 1 kHz (engines, generators)
    psd_db = 10 * np.log10(psd + 1e-30)
    prominence = psd_db - median_filter(psd_db, size=51)        # height above the spectrum within +-50 Hz
    band = np.flatnonzero((fw >= TONAL_RANGE[0]) & (fw <= TONAL_RANGE[1]))
    peaks, props = find_peaks(prominence[band], height=TONAL_MIN_DB, distance=3)
    heights = np.sort(props["peak_heights"])[::-1]
    out["tonal_count"] = len(peaks)
    out["tonal_strongest_hz"] = fw[band][peaks[np.argmax(props["peak_heights"])]] if len(peaks) else 0.0
    out["tonal_strongest_db"] = heights[0] if len(peaks) else 0.0
    out["tonal_top3_db"] = heights[:3].mean() if len(peaks) else 0.0
    near_peak = np.zeros(len(band), dtype=bool)
    for pk in peaks:
        near_peak[max(pk - 1, 0):pk + 2] = True
    out["tonal_power_share"] = psd[band][near_peak].sum() / psd[band].sum()

    # DEMON: envelope of the 2-10 kHz noise; its spectrum shows the propeller's shaft and blade rates
    env = np.sqrt((sosfilt(DEMON_SOS, x) ** 2).reshape(-1, SR // DEMON_RATE).mean(1))
    out["demon_depth"] = env.std() / (env.mean() + 1e-20)
    M = np.abs(np.fft.rfft(detrend(env) * np.hanning(len(env)))) ** 2
    fm = np.fft.rfftfreq(len(env), 1 / DEMON_RATE)
    total = M[(fm >= DEMON_EDGES[0]) & (fm < DEMON_EDGES[-1])].sum() + 1e-30
    for lo, hi in zip(DEMON_EDGES[:-1], DEMON_EDGES[1:]):
        out[f"demon_{lo:.1f}-{hi:.0f}Hz"] = np.log10(M[(fm >= lo) & (fm < hi)].sum() / total + 1e-12)
    in_range = np.flatnonzero((fm >= 1.0) & (fm <= 100.0))
    pk = in_range[np.argmax(M[in_range])]
    out["demon_peak_hz"] = fm[pk]
    out["demon_peak_db"] = 10 * np.log10(M[pk] / (np.median(M[in_range]) + 1e-30))
    return {k: float(v) for k, v in out.items()}


def feature_groups(columns) -> dict[str, list[str]]:
    """Readable group name -> its columns."""
    return {name: [c for c in columns if c.split("_")[0] == prefix] for prefix, name in GROUPS.items()}


def mfcc_columns(columns, n_keep: int) -> list[str]:
    return [c for c in columns if c.startswith("mfcc_") and int(c.split("_")[1]) <= n_keep]


def _features_of_clips(zip_path, names):
    from shipzip import SplitZip
    z = SplitZip(str(zip_path))
    rows = []
    for n in names:
        wave, sr, _ = decode_wav(z.read(n))
        rows.append(engineered_features(fix_length(resample(wave, sr))))
    return rows


def extract_engineered(table, rows, zip_path=AUDIO_ZIP, workers=4, chunk=400) -> pd.DataFrame:
    from joblib import Parallel, delayed
    names = table.zip_name.iloc[rows].tolist()
    parts = [names[s:s + chunk] for s in range(0, len(names), chunk)]
    done = Parallel(n_jobs=workers, verbose=5)(delayed(_features_of_clips)(zip_path, p) for p in parts)
    out = pd.DataFrame([r for part in done for r in part]).astype("float32")
    out.insert(0, "row", rows)
    return out


def build_features(data_dir=DATA_DIR, zip_path=AUDIO_ZIP, workers=4):
    mel, table = D.open_features(data_dir)
    rows = np.flatnonzero(table.label.isin(CLASSES6).to_numpy())
    t0 = time.time()
    simple = pd.DataFrame(features_for(mel, rows, stats=band_stats).astype("float32"),
                          columns=[f"mel{b}_mean" for b in range(128)] + [f"mel{b}_std" for b in range(128)]
                          + ["mel_all_mean"])
    simple.insert(0, "row", rows)
    simple.to_parquet(Path(data_dir) / SIMPLE_FILE)
    print(f"simple features: {simple.shape[1] - 1} per clip, {time.time() - t0:.0f} s")
    t0 = time.time()
    eng = extract_engineered(table, rows, zip_path, workers)
    eng.to_parquet(Path(data_dir) / ENGINEERED_FILE)
    print(f"engineered features: {eng.shape[1] - 1} per clip, {time.time() - t0:.0f} s")


def load_features(data_dir=DATA_DIR):
    """(table, simple, engineered): the two feature tables are indexed by the row number in clips.parquet."""
    table = pd.read_parquet(Path(data_dir) / "clips.parquet")
    simple = pd.read_parquet(Path(data_dir) / SIMPLE_FILE).set_index("row")
    eng = pd.read_parquet(Path(data_dir) / ENGINEERED_FILE).set_index("row")
    return table, simple, eng


# ---------------------------------------------------------------- models

LR_C = (0.01, 0.1, 1.0)
GB_PARAMS = dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0)


def fit_lr(X, y, w, Xva, yva, wva=None, Cs=LR_C):
    """Logistic regression (standardised features); C chosen on the validation ships. Returns (model, C, score)."""
    best = None
    for C in Cs:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        m.fit(X, y, logisticregression__sample_weight=w)
        score = balanced_accuracy_score(yva, m.predict(Xva))
        if best is None or score > best[2]:
            best = (m, C, score)
    return best


def refit_lr(X, y, w, C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(
        X, y, logisticregression__sample_weight=w)


def fit_gb(X, y, w, Xva, yva, wva, seed=0):
    """Gradient boosting; stops adding trees when the validation ships' loss stops improving.
    Returns (model, number of trees, score)."""
    m = HistGradientBoostingClassifier(**GB_PARAMS, max_iter=1000, early_stopping=True, n_iter_no_change=30,
                                       random_state=seed)
    m.fit(X, y, sample_weight=w, X_val=Xva, y_val=yva, sample_weight_val=wva)
    return m, int(m.n_iter_), balanced_accuracy_score(yva, m.predict(Xva))


def refit_gb(X, y, w, n_trees, seed=0):
    return HistGradientBoostingClassifier(**GB_PARAMS, max_iter=n_trees, early_stopping=False,
                                          random_state=seed).fit(X, y, sample_weight=w)


def per_ship_balanced_accuracy(probs, y, ships, n_classes=len(CLASSES6)) -> float:
    """Average the probabilities over all clips of a ship (or background day), one answer per ship; then the mean
    over classes of the share of ships classified correctly."""
    df = pd.DataFrame(probs).assign(y=y, ship=np.asarray(ships))
    per_ship = df.groupby("ship").mean()
    pred, true = per_ship.drop(columns="y").to_numpy().argmax(1), per_ship.y.to_numpy().astype(int)
    return balanced_accuracy_score(true, pred)


def scores(y, probs, ships) -> dict:
    pred = probs.argmax(1)
    recall = confusion_matrix(y, pred, labels=range(len(CLASSES6)), normalize="true").diagonal()
    return {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro"),
            "per_ship_balanced_accuracy": per_ship_balanced_accuracy(probs, y, ships),
            "per_class_recall": dict(zip(CLASSES6, map(float, recall)))}


class Fold:
    """Rows, labels, ships and weights of one fold's train / validation / test split (and train + validation)."""

    def __init__(self, table, folds, k):
        self.k, self.rows = k, split_rows(folds, k)
        self.rows["train+validation"] = np.sort(np.concatenate([self.rows["train"], self.rows["validation"]]))
        index = {c: i for i, c in enumerate(CLASSES6)}
        ships = ship_key(table)
        self.y = {s: table.label.iloc[r].map(index).to_numpy() for s, r in self.rows.items()}
        self.ships = {s: ships.iloc[r].to_numpy() for s, r in self.rows.items()}
        self.w = {s: ship_weights(self.y[s], self.ships[s]) for s in self.rows}

    def X(self, feats: pd.DataFrame, cols, split):
        return feats.loc[self.rows[split], cols].to_numpy()


def fit_and_test(fold: Fold, feats, cols, model, seed=0):
    """Choose the setting on validation, refit on train + validation, predict the test fold.
    Returns (test probabilities, setting, validation score, fitted model)."""
    Xtr, Xva, Xall, Xte = (fold.X(feats, cols, s) for s in ("train", "validation", "train+validation", "test"))
    if model == "logistic regression":
        _, setting, val = fit_lr(Xtr, fold.y["train"], fold.w["train"], Xva, fold.y["validation"])
        final = refit_lr(Xall, fold.y["train+validation"], fold.w["train+validation"], setting)
    else:
        _, setting, val = fit_gb(Xtr, fold.y["train"], fold.w["train"], Xva, fold.y["validation"],
                                 fold.w["validation"], seed)
        final = refit_gb(Xall, fold.y["train+validation"], fold.w["train+validation"], setting, seed)
    return final.predict_proba(Xte), setting, val, final


def validation_score(fold: Fold, feats, cols, model, seed=0) -> float:
    Xtr, Xva = fold.X(feats, cols, "train"), fold.X(feats, cols, "validation")
    if model == "logistic regression":
        return fit_lr(Xtr, fold.y["train"], fold.w["train"], Xva, fold.y["validation"])[2]
    return fit_gb(Xtr, fold.y["train"], fold.w["train"], Xva, fold.y["validation"], fold.w["validation"], seed)[2]


def group_permutation_drop(model, X, y, groups_idx, repeats=3, seed=0) -> dict[str, float]:
    """How much balanced accuracy drops when one feature group is shuffled across clips (all its columns together)."""
    rng = np.random.default_rng(seed)
    base = balanced_accuracy_score(y, model.predict(X))
    drops = {}
    for name, idx in groups_idx.items():
        vals = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, idx] = X[rng.permutation(len(X))][:, idx]
            vals.append(base - balanced_accuracy_score(y, model.predict(Xp)))
        drops[name] = float(np.mean(vals))
    return drops


MODELS = ("logistic regression", "gradient boosting")
MFCC_CHOICES = (4, 8, 12, 16, 19)


def run_all(data_dir=DATA_DIR, out_dir=OUT_DIR, seed=0):
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table, simple, eng = load_features(data_dir)
    folds = make_folds(table, seed=seed)
    pd.DataFrame({"row": np.flatnonzero(folds), "fold": folds[folds > 0]}).to_parquet(out_dir / "folds.parquet")
    fold_list = [Fold(table, folds, k) for k in range(1, N_FOLDS + 1)]

    def log(msg):
        print(f"[{(time.time() - t0) / 60:5.1f} min] {msg}", flush=True)

    # 1. how many MFCCs: MFCC group alone, judged on the validation ships only
    sweep = []
    for n in MFCC_CHOICES:
        cols = mfcc_columns(eng.columns, n)
        for model in MODELS:
            for f in fold_list:
                sweep.append({"n_mfcc": n, "model": model, "fold": f.k,
                              "val_balanced_accuracy": validation_score(f, eng, cols, model, seed)})
        log(f"MFCC 1..{n}: done")
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(out_dir / "mfcc_sweep.csv", index=False)
    mean = sweep.groupby("n_mfcc").val_balanced_accuracy.mean()
    n_mfcc = int(mean.index[mean >= mean.max() - 0.005].min())   # fewest MFCCs within half a point of the best
    log(f"keeping MFCCs 1..{n_mfcc} (validation balanced accuracy by count: {mean.round(3).to_dict()})")

    groups = feature_groups(eng.columns)
    groups["MFCC"] = mfcc_columns(eng.columns, n_mfcc)
    eng_cols = [c for cols in groups.values() for c in cols]
    sets = {"simple (no MFCC)": list(simple.columns), "engineered": eng_cols}
    feats = {"simple (no MFCC)": simple, "engineered": eng}

    # 2. main comparison: two feature sets x two models x five folds, plus group importance for the engineered set
    results, preds, importance = [], [], []
    for f in fold_list:
        for set_name, cols in sets.items():
            for model in MODELS:
                probs, setting, val, final = fit_and_test(f, feats[set_name], cols, model, seed)
                res = scores(f.y["test"], probs, f.ships["test"])
                results.append({"fold": f.k, "features": set_name, "model": model, "n_features": len(cols),
                                "setting": setting, "val_balanced_accuracy": val, **res})
                preds.append(pd.DataFrame(probs, columns=[f"p_{c}" for c in CLASSES6]).assign(
                    row=f.rows["test"], fold=f.k, features=set_name, model=model, y=f.y["test"]))
                if set_name == "engineered":
                    idx = {g: [cols.index(c) for c in gc] for g, gc in groups.items()}
                    drops = group_permutation_drop(final, f.X(feats[set_name], cols, "test"), f.y["test"], idx,
                                                   seed=seed)
                    importance += [{"fold": f.k, "model": model, "group": g, "drop": d} for g, d in drops.items()]
                log(f"fold {f.k}, {set_name}, {model}: balanced accuracy {res['balanced_accuracy']:.3f}")

    # 3. each engineered group on its own
    alone = []
    for g, cols in groups.items():
        for model in MODELS:
            for f in fold_list:
                probs, *_ = fit_and_test(f, eng, cols, model, seed)
                alone.append({"group": g, "model": model, "fold": f.k, "n_features": len(cols),
                              **{k: v for k, v in scores(f.y["test"], probs, f.ships["test"]).items()
                                 if k != "per_class_recall"}})
        log(f"group alone: {g}")

    pd.concat(preds, ignore_index=True).to_parquet(out_dir / "predictions.parquet")
    pd.DataFrame(importance).to_csv(out_dir / "group_importance.csv", index=False)
    pd.DataFrame(alone).to_csv(out_dir / "group_alone.csv", index=False)
    summary = {"classes": CLASSES6, "n_mfcc": n_mfcc, "feature_sets": {k: len(v) for k, v in sets.items()},
               "engineered_groups": {g: len(c) for g, c in groups.items()}, "seed": seed,
               "minutes": round((time.time() - t0) / 60, 1), "results": results}
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    log("finished")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["features", "run"])
    p.add_argument("--workers", type=int, default=4, help="processes reading the audio (features step)")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = p.parse_args(argv)
    with keep_awake():
        if a.step == "features":
            build_features(a.data_dir, workers=a.workers)
        else:
            summary = run_all(a.data_dir, a.out_dir)
            df = pd.DataFrame(summary["results"])
            print(df.groupby(["features", "model"])[["accuracy", "balanced_accuracy", "macro_f1",
                                                     "per_ship_balanced_accuracy"]].agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
