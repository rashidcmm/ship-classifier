"""Six classes balanced by removing data instead of weighting it (lesson 14).

Lesson 13 kept every clip and gave each class and each ship the same training weight. Here no weights are used.
Instead, the training data are cut until every class has
  - the same number of ships: as many as the class with the fewest ships in that run (tanker, about 187), chosen at
    random, and
  - the same number of clips: as many as the class with the fewest clips among the kept ships (cargo, whose ships have
    1-2 clips each), taken evenly: first one clip from every kept ship, then a second one, ...
The folds, test sets, features and models are those of lesson 13, so the two ways of balancing are compared on exactly
the same test clips. The test folds are never cut. Which ships are removed is random, so every run is repeated with
several random draws.

A team of balanced models (--ensemble): in every run, many models are trained, each on its own random equal-ships
subset of the same training folds, and their probabilities are averaged on the (same) test fold. Every model is
balanced, but together they see most of the training ships.

  python src/six_classes_equal.py              # -> runs/six_classes_equal/ (a few minutes, CPU)
  python src/six_classes_equal.py --ensemble   # -> runs/six_classes_equal/ensemble_* (about 70 minutes, CPU)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import six_classes as S
from keepawake import keep_awake
from paths import DATA_DIR, RUNS_DIR

OUT_DIR = RUNS_DIR / "six_classes_equal"
N_DRAWS = 5
N_MEMBERS = 25
CHECKPOINTS = (1, 2, 3, 5, 10, 15, 20, 25)           # team sizes that are scored


def balance_by_removal(labels, ships, seed=0) -> np.ndarray:
    """Positions of the clips to keep: the same number of ships in every class, then the same number of clips."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"label": np.asarray(labels), "ship": np.asarray(ships)})
    n_ships = df.groupby("label").ship.nunique().min()
    kept = np.concatenate([rng.choice(g.ship.unique(), n_ships, replace=False) for _, g in df.groupby("label")])
    sub = df[df.ship.isin(kept)].assign(order=lambda d: rng.random(len(d)))
    sub["turn"] = sub.groupby("ship")["order"].rank(method="first")   # 1 for each ship's first clip, 2, ...
    n_clips = sub.groupby("label").size().min()
    keep = sub.sort_values(["turn", "order"]).groupby("label").head(n_clips).index
    return np.sort(keep.to_numpy())


class EqualFold:
    """A lesson-13 fold whose train, validation and train + validation sets are cut to equal ships and clips per
    class. The test set is left whole. No weights."""

    def __init__(self, fold: S.Fold, seed: int):
        self.k, self.rows, self.y, self.ships = fold.k, {}, {}, {}
        for split in fold.rows:
            keep = (np.arange(len(fold.rows[split])) if split == "test"
                    else balance_by_removal(fold.y[split], fold.ships[split], seed))
            self.rows[split] = fold.rows[split][keep]
            self.y[split], self.ships[split] = fold.y[split][keep], fold.ships[split][keep]
        self.w = {split: None for split in self.rows}

    X = S.Fold.X

    def counts(self, split="train+validation") -> pd.DataFrame:
        df = pd.DataFrame({"label": np.array(S.CLASSES6)[self.y[split]], "ship": self.ships[split]})
        return df.groupby("label").agg(ships=("ship", "nunique"), clips=("ship", "size"))


def _setup(data_dir, n_mfcc):
    """Clip table, feature sets (as in lesson 13), the tables they come from, and lesson 13's folds."""
    table, simple, eng = S.load_features(data_dir)
    if n_mfcc is None:                                   # the number of MFCCs chosen in lesson 13
        lesson13 = S.OUT_DIR / "results.json"
        n_mfcc = json.loads(lesson13.read_text())["n_mfcc"] if lesson13.exists() else S.N_MFCC
    groups = S.feature_groups(eng.columns)
    groups["MFCC"] = S.mfcc_columns(eng.columns, n_mfcc)
    sets = {"simple (no MFCC)": list(simple.columns), "engineered": [c for g in groups.values() for c in g]}
    feats = {"simple (no MFCC)": simple, "engineered": eng}
    return table, sets, feats, S.make_folds(table), n_mfcc


def run_equal(data_dir=DATA_DIR, out_dir=OUT_DIR, draws=N_DRAWS, n_mfcc=None):
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table, sets, feats, folds, n_mfcc = _setup(data_dir, n_mfcc)

    results, preds, counts = [], [], []
    for k in range(1, S.N_FOLDS + 1):
        full = S.Fold(table, folds, k)
        for draw in range(draws):
            f = EqualFold(full, seed=100 * k + draw)
            for split in ("train", "validation", "train+validation"):
                c = f.counts(split)
                counts += [{"fold": k, "draw": draw, "split": split, "label": lab, **row}
                           for lab, row in c.to_dict("index").items()]
            for set_name, cols in sets.items():
                for model in S.MODELS:
                    probs, setting, val, _ = S.fit_and_test(f, feats[set_name], cols, model)
                    res = S.scores(f.y["test"], probs, f.ships["test"])
                    results.append({"fold": k, "draw": draw, "features": set_name, "model": model,
                                    "n_features": len(cols), "n_train_clips": int(len(f.rows["train+validation"])),
                                    "setting": setting, "val_balanced_accuracy": val, **res})
                    if draw == 0:
                        preds.append(pd.DataFrame(probs, columns=[f"p_{c}" for c in S.CLASSES6]).assign(
                            row=f.rows["test"], fold=k, features=set_name, model=model, y=f.y["test"]))
            print(f"[{(time.time() - t0) / 60:4.1f} min] fold {k}, draw {draw}: "
                  f"{len(f.rows['train+validation'])} training clips", flush=True)

    pd.concat(preds, ignore_index=True).to_parquet(out_dir / "predictions.parquet")
    pd.DataFrame(counts).to_csv(out_dir / "counts.csv", index=False)
    summary = {"classes": S.CLASSES6, "n_mfcc": n_mfcc, "draws": draws,
               "feature_sets": {k: len(v) for k, v in sets.items()},
               "minutes": round((time.time() - t0) / 60, 1), "results": results}
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_ensemble(data_dir=DATA_DIR, out_dir=OUT_DIR, members=N_MEMBERS, checkpoints=CHECKPOINTS, n_mfcc=None):
    """In every run, train `members` models, each on its own equal-ships subset (member i uses the same random draw
    as draw i of run_equal), and average their test probabilities. Scores and the share of training ships and clips
    seen by at least one member are recorded after 1, 2, 3, 5, ... members."""
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table, sets, feats, folds, n_mfcc = _setup(data_dir, n_mfcc)
    checkpoints = [m for m in checkpoints if m <= members]

    results, coverage, preds = [], [], []
    for k in range(1, S.N_FOLDS + 1):
        full = S.Fold(table, folds, k)
        tv = pd.DataFrame({"label": np.array(S.CLASSES6)[full.y["train+validation"]],
                           "ship": full.ships["train+validation"], "row": full.rows["train+validation"]})
        total = {(f, m): 0.0 for f in sets for m in S.MODELS}
        seen = np.zeros(len(tv), dtype=bool)
        for i in range(members):
            f = EqualFold(full, seed=100 * k + i)
            seen |= np.isin(tv.row.to_numpy(), f.rows["train+validation"])
            for set_name, cols in sets.items():
                for model in S.MODELS:
                    total[(set_name, model)] = total[(set_name, model)] + S.fit_and_test(f, feats[set_name], cols,
                                                                                       model)[0]
            if i + 1 in checkpoints:
                for (set_name, model), summed in total.items():
                    res = S.scores(full.y["test"], summed / (i + 1), full.ships["test"])
                    results.append({"fold": k, "members": i + 1, "features": set_name, "model": model, **res})
                marked = tv.assign(seen=seen)
                ships_seen = marked.groupby(["label", "ship"]).seen.any().groupby("label").mean()
                clips_seen = marked.groupby("label").seen.mean()
                coverage += [{"fold": k, "members": i + 1, "label": lab, "ships_seen": ships_seen[lab],
                              "clips_seen": clips_seen[lab]} for lab in S.CLASSES6]
            print(f"[{(time.time() - t0) / 60:5.1f} min] fold {k}: {i + 1} members", flush=True)
        for (set_name, model), summed in total.items():
            preds.append(pd.DataFrame(summed / members, columns=[f"p_{c}" for c in S.CLASSES6]).assign(
                row=full.rows["test"], fold=k, features=set_name, model=model, y=full.y["test"]))

    pd.concat(preds, ignore_index=True).to_parquet(out_dir / "ensemble_predictions.parquet")
    pd.DataFrame(coverage).to_csv(out_dir / "ensemble_coverage.csv", index=False)
    summary = {"classes": S.CLASSES6, "n_mfcc": n_mfcc, "members": members, "checkpoints": checkpoints,
               "minutes": round((time.time() - t0) / 60, 1), "results": results}
    (out_dir / "ensemble_results.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--draws", type=int, default=N_DRAWS, help="random choices of the ships and clips to keep")
    p.add_argument("--ensemble", action="store_true", help="a team of balanced models instead of single ones")
    p.add_argument("--members", type=int, default=N_MEMBERS, help="models in the team (--ensemble)")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = p.parse_args(argv)
    with keep_awake():
        summary = (run_ensemble(a.data_dir, a.out_dir, a.members) if a.ensemble
                   else run_equal(a.data_dir, a.out_dir, a.draws))
    df = pd.DataFrame(summary["results"])
    keys = ["features", "model"] + (["members"] if a.ensemble else [])
    print(df.groupby(keys)[["balanced_accuracy", "per_ship_balanced_accuracy"]].agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
