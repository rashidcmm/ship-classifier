"""Every main classical result again, with an "other" class always in (lessons 22-24).

Lessons 13-20 left six ship types out: sailing, other, passenger ship, rescue, dredger and pilot vessel (about 520
ships, 30 h of audio). A system in the water hears them too, so from here on every grouping keeps them, as one class
"other" (lesson 21, section A). Same folds (open_set.all_folds: lesson 13's folds for the six classes, five folds by
ship for the left-out ones), same weights (every class the same total; inside a merged class every original type the
same; inside that every ship), same protocol (settings on the validation fold, refit on training + validation, the
test fold scored once, per clip and per ship).

  python src/with_other.py features   # mel-band statistics of all 43,208 clips -> data/features_all_mel.parquet
  python src/with_other.py main       # lesson 22: 6 classes + other; five feature sets x LR, GB x 5 folds
  python src/with_other.py merged     # lesson 24: 6/5/4 classes + other with the chosen set; PANN embeddings
  python src/with_other.py cost       # lesson 23: cost of every feature set; sets added one at a time, kept if better
  python src/with_other.py bands      # lesson 23: 128 mel bands merged to 64/32/16, or only the lowest 20/30/40/64
  python src/with_other.py all        # everything above, in this order (CPU, results in runs/with_other/)
  python src/with_other.py main merged   &   python src/with_other.py cost bands     # the same in two processes, ~1.5 h

Every finished run is appended to runs/with_other/<step>.jsonl at once, so a stopped job resumes where it was.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D
import features as F
import merged_classes as M
import open_set as O
import panns_features as P
import six_classes as S
from keepawake import keep_awake
from paths import AUDIO_ZIP, DATA_DIR, RUNS_DIR

OUT_DIR = RUNS_DIR / "with_other"
MEL_FILE = "features_all_mel.parquet"
LEFT_OUT = O.LEFT_OUT
SIX = {c: [c] for c in S.CLASSES6}
GROUPINGS = {"6 classes + other": {**SIX, "other": LEFT_OUT},
             "5 classes + other": {**O.FIVE, "other": LEFT_OUT},
             "4 classes + other": {**O.FOUR, "other": LEFT_OUT}}
MAIN = "6 classes + other"
MODELS = S.MODELS                     # ("logistic regression", "gradient boosting")
THREADS = 4                           # maths-library threads: more is slower on this laptop (lesson 15)
TOLERANCE = 0.005                     # "as good as": within half a point of the best validation score
PERCENTILES = (10, 50, 90)
MERGE = (2, 4, 8)                     # neighbouring mel bands merged: 64, 32, 16 bands
LOWEST = (20, 30, 40, 64)             # only the lowest n of the 128 mel bands
SCREEN_C = 0.1                        # cost step: logistic regression with one C judges each step
SCREEN_FOLDS = (1, 2, 3, 4, 5)        # ... on the validation folds of all five rounds (as lesson 15)
N_TIMING_CLIPS = 120
N_PCA = O.N_PCA
SEP = " | "                            # joins the group names of one combination
# band b of the 128-band mel spectrogram is a triangle from MEL_EDGES[b] to MEL_EDGES[b + 2]
MEL_EDGES = F.mel_to_hz(np.linspace(F.hz_to_mel(F.FMIN), F.hz_to_mel(F.FMAX), F.N_MELS + 2))


# ---------------------------------------------------------------- features

def mel_statistics(mel_block) -> pd.DataFrame:
    """Clips of the cached log-mel spectrogram (dB, (clips, 128, 501)) -> per band mean, spread and 10/50/90th
    percentile, the overall mean, and mean + spread of neighbouring bands merged by 2, 4 and 8 (power summed)."""
    x = np.asarray(mel_block, dtype=np.float32)
    n, nb = len(x), x.shape[1]
    out = {}
    mean, std = x.mean(axis=2), x.std(axis=2)
    for b in range(nb):
        out[f"mel{b}_mean"] = mean[:, b]
    for b in range(nb):
        out[f"mel{b}_std"] = std[:, b]
    out["mel_all_mean"] = x.mean(axis=(1, 2))
    pct = np.percentile(x, PERCENTILES, axis=2)
    for i, p in enumerate(PERCENTILES):
        for b in range(nb):
            out[f"mel{b}_p{p}"] = pct[i, :, b]
    power = 10.0 ** (x / 10.0)
    for m in MERGE:
        g = nb // m
        db = 10 * np.log10(power[:, :g * m].reshape(n, g, m, -1).sum(axis=2) + F.POWER_FLOOR)
        mm, ss = db.mean(axis=2), db.std(axis=2)
        for j in range(g):
            out[f"band{m}_{j}_mean"], out[f"band{m}_{j}_std"] = mm[:, j], ss[:, j]
    return pd.DataFrame(out).astype("float32")


def build_mel_features(data_dir=DATA_DIR, chunk=250) -> Path:
    t0 = time.time()
    mel, table = D.open_features(data_dir)
    rows = np.flatnonzero(table.label.isin(S.CLASSES6 + LEFT_OUT).to_numpy())
    parts = [mel_statistics(mel[rows[s:s + chunk]]) for s in range(0, len(rows), chunk)]
    df = pd.concat(parts, ignore_index=True)
    df.insert(0, "row", rows)
    path = Path(data_dir) / MEL_FILE
    df.to_parquet(path)
    print(f"mel statistics: {df.shape[1] - 1} columns for {len(df):,} clips, {time.time() - t0:.0f} s", flush=True)
    return path


def load_all(data_dir=DATA_DIR):
    """(table, folds, features): engineered (144) + mel statistics for every clip of the twelve labels."""
    table = pd.read_parquet(Path(data_dir) / "clips.parquet")
    eng = O.load_engineered(data_dir)
    mel = pd.read_parquet(Path(data_dir) / MEL_FILE).set_index("row")
    return table, O.all_folds(table), eng.join(mel, how="inner"), list(eng.columns)


def mean_spread_cols(n_bands: int = F.N_MELS, overall: bool = True) -> list[str]:
    cols = [f"mel{b}_mean" for b in range(n_bands)] + [f"mel{b}_std" for b in range(n_bands)]
    return cols + ["mel_all_mean"] if overall else cols


def percentile_cols() -> list[str]:
    return [f"mel{b}_p{p}" for p in PERCENTILES for b in range(F.N_MELS)]


def feature_sets(eng_cols) -> dict[str, list[str]]:
    """Lesson 22's five sets: the mel bands before any MFCC (mean + spread, then + percentiles), the 144 engineered
    numbers, and the two together."""
    ms, pc = mean_spread_cols(), percentile_cols()
    sets = {"mel mean + spread": ms, "mel mean + spread + percentiles": ms + pc, "engineered": list(eng_cols),
            "engineered + mel mean + spread": list(eng_cols) + ms,
            "engineered + mel mean + spread + percentiles": list(eng_cols) + ms + pc}
    return {f"{k} ({len(v)})": v for k, v in sets.items()}


def top_hz_of_lowest(n: int) -> float:
    """Upper edge of the n-th mel band: the lowest n bands cover 10 Hz up to this frequency."""
    return float(MEL_EDGES[n + 1])


def band_variants() -> dict[str, list[str]]:
    """Lesson 23: mean + spread of every band, with the 128 bands merged or only the lowest ones kept."""
    v = {"128 bands, 10 Hz-16 kHz": mean_spread_cols(overall=False)}
    for m in MERGE:
        g = F.N_MELS // m
        v[f"{g} bands (neighbours merged by {m}), 10 Hz-16 kHz"] = (
            [f"band{m}_{j}_mean" for j in range(g)] + [f"band{m}_{j}_std" for j in range(g)])
    for n in LOWEST:
        v[f"lowest {n} bands, 10 Hz-{top_hz_of_lowest(n):,.0f} Hz"] = mean_spread_cols(n, overall=False)
    return v


def cost_groups(eng_cols) -> dict[str, list[str]]:
    """The nine feature sets of lesson 23: two from the mel spectrogram, seven engineered."""
    return {"mel mean + spread": mean_spread_cols(), "mel percentiles": percentile_cols(),
            **S.feature_groups(eng_cols)}


# ---------------------------------------------------------------- what the features cost to compute

def _log_mel_128(wave: np.ndarray) -> np.ndarray:
    """The cached spectrogram's recipe in numpy (4,096-sample window, 10 ms hop, 128 mel bands, dB)."""
    x = np.pad(wave.astype(np.float32), F.N_FFT // 2, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(x, F.N_FFT)[::F.HOP]
    power = np.abs(np.fft.rfft(frames * np.hanning(F.N_FFT).astype(np.float32), axis=1)) ** 2
    return 10 * np.log10(power @ _MEL_FB.T + F.POWER_FLOOR).T


_MEL_FB = F.mel_filterbank().astype(np.float32)
# what each group needs computed first (shared with other groups, paid once per clip)
PREREQ = {"mel mean + spread": "mel spectrogram (128 bands)", "mel percentiles": "mel spectrogram (128 bands)",
          "MFCC": "STFT (2,048 window)", "level changes over time": "STFT (2,048 window)",
          "spectral shape": "STFT (2,048 window)", "1/3-octave levels": "Welch spectrum",
          "loudness and impulsiveness": "Welch spectrum", "tonal lines": "Welch spectrum",
          "propeller modulation (DEMON)": None}


def measure_costs(table, rows, zip_path=AUDIO_ZIP) -> dict[str, float]:
    """Milliseconds per clip, one CPU thread, for reading the clip, each shared step and each group on its own."""
    from shipzip import SplitZip
    z = SplitZip(str(zip_path))
    times = {}

    def tick(name, fn, *a):
        t = time.perf_counter()
        r = fn(*a)
        times.setdefault(name, []).append(1000 * (time.perf_counter() - t))
        return r

    prefix_name = S.GROUPS
    with threadpool_limits(1):
        for r in rows:
            raw = tick("read and decode", z.read, table.zip_name.iloc[r])
            wave = tick("read and decode", lambda b: F.fix_length(F.resample(*F.decode_wav(b)[:2])), raw)
            logmel = tick("mel spectrogram (128 bands)", _log_mel_128, wave)[None]
            tick("mel mean + spread", lambda m: (m.mean(axis=2), m.std(axis=2), m.mean()), logmel)
            tick("mel percentiles", lambda m: np.percentile(m, PERCENTILES, axis=2), logmel)
            sp = tick("STFT (2,048 window)", S._spectra, wave)
            sp = tick("Welch spectrum", S._welch, sp)
            for prefix, (fn, _) in S.GROUP_FUNCTIONS.items():
                tick(prefix_name[prefix], fn, sp, {})
    # "read and decode" was ticked twice per clip (read, then decode): add the two
    out = {k: float(np.median(v)) for k, v in times.items() if k != "read and decode"}
    rd = np.array(times["read and decode"]).reshape(-1, 2).sum(1)
    out["read and decode"] = float(np.median(rd))
    return out


def subset_cost(groups, costs) -> float:
    """Milliseconds per clip to compute these groups: reading the clip + every needed shared step once + each group."""
    shared = {PREREQ[g] for g in groups if PREREQ[g]}
    return costs["read and decode"] + sum(costs[s] for s in shared) + sum(costs[g] for g in groups)


# ---------------------------------------------------------------- running

class Log:
    """Results of one step, one JSON line per finished run, so a stopped job resumes where it was. Two processes may
    share a log (lesson 22's sets, from both ends): the file is read again whenever it changed."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.rows, self._mtime = [], None
        self._refresh()

    def _refresh(self):
        if self.path.exists() and self.path.stat().st_mtime != self._mtime:
            self._mtime = self.path.stat().st_mtime
            self.rows = [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def has(self, **key) -> bool:
        self._refresh()
        return any(all(r.get(k) == v for k, v in key.items()) for r in self.rows)

    def add(self, rec: dict, key: dict | None = None) -> bool:
        """Append the run; with a key, only if no other process logged the same run in the meantime."""
        if key and self.has(**key):
            return False
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        self._refresh()
        return True


def fit_and_test(f, feats, cols, model, seed=0):
    """lesson 13's protocol (six_classes.fit_and_test), timed: (test probabilities, setting, validation score,
    seconds to choose the setting, seconds for the final fit, microseconds per clip to predict)."""
    Xtr, Xva, Xall, Xte = (f.X(feats, cols, s) for s in ("train", "validation", "train+validation", "test"))
    t0 = time.perf_counter()
    if model == "logistic regression":
        _, setting, val = S.fit_lr(Xtr, f.y["train"], f.w["train"], Xva, f.y["validation"])
        t1 = time.perf_counter()
        final = S.refit_lr(Xall, f.y["train+validation"], f.w["train+validation"], setting)
    else:
        _, setting, val = S.fit_gb(Xtr, f.y["train"], f.w["train"], Xva, f.y["validation"], f.w["validation"], seed)
        t1 = time.perf_counter()
        final = S.refit_gb(Xall, f.y["train+validation"], f.w["train+validation"], setting, seed)
    t2 = time.perf_counter()
    probs = final.predict_proba(Xte)
    t3 = time.perf_counter()
    return probs, setting, val, t1 - t0, t2 - t1, 1e6 * (t3 - t2) / len(Xte)


class Runner:
    def __init__(self, table, folds, feats, out_dir, seed=0, t0=None):
        self.table, self.folds, self.feats, self.seed = table, folds, feats, seed
        self.out_dir = Path(out_dir)
        self.t0 = t0 or time.time()
        self._folds = {}

    def log(self, msg):
        print(f"[{(time.time() - self.t0) / 60:6.1f} min] {msg}", flush=True)

    def fold(self, grouping_name, k) -> O.GroupFold:
        key = (grouping_name, k)
        if key not in self._folds:
            self._folds[key] = O.GroupFold(self.table, self.folds, k, GROUPINGS[grouping_name])
        return self._folds[key]

    def run(self, step: str, log: Log, grouping: str, features: str, cols, model, k, feats=None, **extra):
        key = dict(grouping=grouping, features=features, model=model, fold=k)
        if log.has(**key):
            return
        f = self.fold(grouping, k)
        probs, setting, val, t_choose, t_fit, us = fit_and_test(f, self.feats if feats is None else feats, cols,
                                                                model, self.seed)
        names = list(GROUPINGS[grouping])
        rec = {**key, "n_features": len(cols), "setting": setting, "val_balanced_accuracy": float(val),
               "seconds_choose": t_choose, "seconds_fit": t_fit, "predict_us_per_clip": us, **extra,
               **M.scores(f.y["test"], probs, f.ships["test"], names)}
        parts = self.out_dir / "parts" / step
        parts.mkdir(parents=True, exist_ok=True)
        slug = f"{grouping}|{features}|{model}|{k}".replace(" ", "_").replace("/", "-").replace("+", "p")
        pd.DataFrame(probs.astype("float32"), columns=[f"p_{n}" for n in names]).assign(
            row=f.rows["test"], fold=k, y=f.y["test"], label=f.label["test"], ship=f.ships["test"],
            grouping=grouping, features=features, model=model).to_parquet(parts / f"{hashlib.md5(slug.encode()).hexdigest()[:16]}.parquet")
        if not log.add(rec, key):
            return
        self.log(f"{step}: {grouping} | {features} | {model} | fold {k}: {rec['balanced_accuracy']:.3f} per clip, "
                 f"{rec['per_ship_balanced_accuracy']:.3f} per ship, other {rec['per_class_recall'].get('other', 0):.3f}")


def collect(out_dir: Path, step: str) -> None:
    """All of a step's prediction parts -> one file (runs/with_other/<step>_predictions.parquet)."""
    parts = sorted((Path(out_dir) / "parts" / step).glob("*.parquet"))
    if parts:
        pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).to_parquet(
            Path(out_dir) / f"{step}_predictions.parquet")


def chosen_set(out_dir: Path, sets: dict) -> str:
    """The set used from lesson 24 on: the best mean validation score (gradient boosting, 6 classes + other); among
    the sets within half a point of it, the one with the fewest numbers."""
    res = pd.DataFrame(Log(Path(out_dir) / "main.jsonl").rows)
    if (res.model == "gradient boosting").any():
        res = res[res.model == "gradient boosting"]
    gb = res.groupby("features").val_balanced_accuracy.mean()
    ok = gb[gb >= gb.max() - TOLERANCE].index
    return min(ok, key=lambda s: len(sets[s]))


# ---------------------------------------------------------------- steps

def step_main(r: Runner, eng_cols, reverse=False):
    log = Log(r.out_dir / "main.jsonl")
    items = list(feature_sets(eng_cols).items())
    for name, cols in (items[::-1] if reverse else items):
        for model in MODELS:
            for k in range(1, S.N_FOLDS + 1):
                r.run("main", log, MAIN, name, cols, model, k)
    collect(r.out_dir, "main")


def step_bands(r: Runner):
    log = Log(r.out_dir / "bands.jsonl")
    for name, cols in band_variants().items():
        for model in MODELS:
            for k in range(1, S.N_FOLDS + 1):
                r.run("bands", log, MAIN, name, cols, model, k)
    collect(r.out_dir, "bands")


def screen(f: O.GroupFold, feats, cols) -> float:
    """Validation balanced accuracy of one logistic regression (C fixed) trained on the training folds."""
    m = make_pipeline(StandardScaler(), LogisticRegression(C=SCREEN_C, max_iter=2000))
    m.fit(f.X(feats, cols, "train"), f.y["train"], logisticregression__sample_weight=f.w["train"])
    return balanced_accuracy_score(f.y["validation"], m.predict(f.X(feats, cols, "validation")))


def group_cols(groups: dict, combo: str) -> list[str]:
    return [c for g in combo.split(SEP) for c in groups[g]]


def step_cost(r: Runner, eng_cols):
    """Lesson 23: one feature set added at a time. The nine sets are ordered by their own validation score (best
    first). Start with the best one; add the next set: if the validation score goes up, keep it, otherwise leave it
    out and try the following set on top of the last kept one. Every step records the numbers per clip and the
    milliseconds per clip to compute them, so accuracy can be weighed against cost."""
    groups = cost_groups(eng_cols)
    cost_path = r.out_dir / "costs.json"
    if cost_path.exists():
        costs = json.loads(cost_path.read_text())
    else:
        rng = np.random.default_rng(r.seed)
        rows = np.sort(rng.choice(np.flatnonzero(r.folds > 0), N_TIMING_CLIPS, replace=False))
        costs = measure_costs(r.table, rows)
        cost_path.write_text(json.dumps(costs, indent=2))
    r.log("costs (ms per clip, one thread): " + ", ".join(f"{k} {v:.1f}" for k, v in costs.items()))

    log = Log(r.out_dir / "walk.jsonl")

    def score(combo: str, stage: str, **extra) -> float:
        done = [x for x in log.rows if x["stage"] == stage and x["groups"] == combo]
        if done:
            return done[0]["val"]
        cols = group_cols(groups, combo)
        t = time.perf_counter()
        vals = [screen(r.fold(MAIN, k), r.feats, cols) for k in SCREEN_FOLDS]
        log.add({"stage": stage, "groups": combo, "n_features": len(cols), "ms": subset_cost(combo.split(SEP), costs),
                 "val": float(np.mean(vals)), "val_by_fold": vals, "seconds": time.perf_counter() - t, **extra})
        return float(np.mean(vals))

    alone = {g: score(g, "alone") for g in groups}
    order = sorted(alone, key=alone.get, reverse=True)
    r.log("order (validation score alone): " + ", ".join(f"{g} {alone[g]:.3f}" for g in order))
    kept, best = [order[0]], alone[order[0]]
    for step, g in enumerate(order[1:], start=2):
        trial = SEP.join(kept + [g])
        val = score(trial, "walk", step=step, added=g, before=best)
        r.log(f"step {step}: + {g}: {val:.3f} ({100 * (val - best):+.2f} points) -> {'kept' if val > best else 'left out'}")
        if val > best:
            kept, best = kept + [g], val
    final = SEP.join(kept)
    (r.out_dir / "walk_result.json").write_text(json.dumps({"order": order, "alone": alone, "final": final,
                                                            "final_val": best}, indent=2))

    # tested like every other set: the walk's final set, and the best single set for reference
    test_log = Log(r.out_dir / "cost.jsonl")
    for combo in dict.fromkeys([order[0], final]):
        for model in MODELS:
            for k in range(1, S.N_FOLDS + 1):
                r.run("cost", test_log, MAIN, combo, group_cols(groups, combo), model, k,
                      ms=subset_cost(combo.split(SEP), costs))
    collect(r.out_dir, "cost")


def step_merged(r: Runner, eng_cols, data_dir=DATA_DIR):
    sets = feature_sets(eng_cols)
    name = chosen_set(r.out_dir, sets)
    r.log(f"chosen set (main step, validation): {name}")
    log = Log(r.out_dir / "merged.jsonl")
    for grouping in GROUPINGS:
        for model in MODELS:
            for k in range(1, S.N_FOLDS + 1):
                r.run("merged", log, grouping, name, sets[name], model, k)

    # PANN embeddings (CNN14, AudioSet), 2,048 -> 256 by PCA fitted on the training + validation clips of each round
    emb = P.load_embeddings(data_dir)
    for k in range(1, S.N_FOLDS + 1):
        f = r.fold(MAIN, k)
        todo = [(fs, m) for fs in ("PANN (256)", f"PANN (256) + {name}") for m in MODELS
                if not log.has(grouping=MAIN, features=fs, model=m, fold=k)]
        if not todo:
            continue
        pca = PCA(N_PCA, random_state=r.seed).fit(emb.loc[f.rows["train+validation"]].to_numpy(np.float32))
        use = np.concatenate([f.rows["train+validation"], f.rows["test"]])
        z = pd.DataFrame(pca.transform(emb.loc[use].to_numpy(np.float32)), index=use,
                         columns=[f"pann_pc{i}" for i in range(N_PCA)])
        feats = z.join(r.feats.loc[use, sets[name]])
        for fs, m in todo:
            cols = list(z.columns) + (sets[name] if fs != "PANN (256)" else [])
            r.run("merged", log, MAIN, fs, cols, m, k, feats=feats,
                  pca_variance=float(pca.explained_variance_ratio_.sum()))
    collect(r.out_dir, "merged")


STEPS = ("main", "merged", "cost", "bands")


def run(steps, data_dir=DATA_DIR, out_dir=OUT_DIR, seed=0, reverse=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(THREADS):
        table, folds, feats, eng_cols = load_all(data_dir)
        r = Runner(table, folds, feats, out_dir, seed)
        r.log(f"{len(feats):,} clips, {feats.shape[1]} columns; steps: {', '.join(steps)}")
        for step in steps:
            if step == "main":
                step_main(r, eng_cols, reverse)
            elif step == "bands":
                step_bands(r)
            elif step == "cost":
                step_cost(r, eng_cols)
            elif step == "merged":
                step_merged(r, eng_cols, data_dir)
        r.log("finished")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", nargs="+", choices=["features", *STEPS, "all"],
                   help="one or more steps; two processes can run different steps at the same time (own logs)")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--reverse", action="store_true", help="main step: the feature sets from the last one (a second process)")
    a = p.parse_args(argv)
    steps = STEPS if "all" in a.step else tuple(st for st in a.step if st != "features")
    with keep_awake():
        if "features" in a.step or not (a.data_dir / MEL_FILE).exists():
            build_mel_features(a.data_dir)
        if steps:
            run(steps, a.data_dir, a.out_dir, reverse=a.reverse)


if __name__ == "__main__":
    main()
