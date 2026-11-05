"""Lesson 27: lesson 23's walk again, with its four weak points fixed (6 classes + other).

Lesson 23 added nine feature sets one at a time and kept a set whenever the validation score rose at all. Four weak
points (questions for review 3, 5 November), and what this script does about each:

1. Tiny gains are noise. A set is kept only if it adds at least MIN_GAIN (0.5 points) on average over the five
   validation folds AND is better in at least MIN_FOLDS (3) of the 5 folds. Paired: the same folds with and without it.
2. A choice is never revisited. After every kept set, each earlier set is tried out again: a set that no longer earns
   its place by the same rule is dropped (floating forward selection).
3. The judge was logistic regression, the reported model gradient boosting. Here gradient boosting judges as well
   (lesson 13's settings, trees stopped on the validation ships). Logistic regression runs again with the new rules,
   so the effect of the rules and the effect of the judge can be told apart.
4. The walk is greedy. After it, every set left out is tried once more on top of the final set, then the left-out
   sets in pairs (two sets might only help together). Still no search over all combinations.

Every validation score is logged (runs/feature_walk2/steps.jsonl), so a stopped job resumes. The final set of each
judge is then tested like every other set: settings on validation, refit on training + validation, the test fold
scored once, both models, 5 folds (runs/feature_walk2/test.jsonl).

  python src/feature_walk2.py                # logistic regression judge (~1 h), gradient boosting judge (~5 h), tests
  python src/feature_walk2.py --judge lr     # one judge only (still tests its final set)
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import six_classes as S
import with_other as W
from keepawake import keep_awake
from paths import DATA_DIR, RUNS_DIR

OUT_DIR = RUNS_DIR / "feature_walk2"
MIN_GAIN = 0.005          # a set must add at least half a point (mean over the five validation folds) ...
MIN_FOLDS = 3             # ... and be better in at least 3 of the 5 folds
JUDGES = {"lr": "logistic regression", "gb": "gradient boosting"}


def beats(new: np.ndarray, old: np.ndarray) -> tuple[bool, float, int]:
    """Is per-fold score `new` better than `old` by the rule? Returns (yes/no, mean gain, folds better)."""
    d = np.asarray(new) - np.asarray(old)
    return bool(d.mean() >= MIN_GAIN and (d > 0).sum() >= MIN_FOLDS), float(d.mean()), int((d > 0).sum())


class Walk:
    def __init__(self, r: W.Runner, groups: dict, judge: str, log: W.Log, costs: dict):
        self.r, self.groups, self.judge, self.log, self.costs = r, groups, judge, log, costs
        self.names = list(groups)              # fixed order: one spelling for every combination
        self.history = []

    def canon(self, combo) -> str:
        return W.SEP.join(sorted(set(combo), key=self.names.index))

    def score(self, combo) -> np.ndarray:
        """Validation balanced accuracy on each of the five folds (trained on the training folds only)."""
        key = self.canon(combo)
        done = [x for x in self.log.rows if x["judge"] == self.judge and x["groups"] == key]
        if done:
            return np.array(done[0]["val_by_fold"])
        cols = W.group_cols(self.groups, key)
        t, vals, trees = time.perf_counter(), [], []
        for k in W.SCREEN_FOLDS:
            f = self.r.fold(W.MAIN, k)
            if self.judge == "gb":
                _, n, v = S.fit_gb(f.X(self.r.feats, cols, "train"), f.y["train"], f.w["train"],
                                   f.X(self.r.feats, cols, "validation"), f.y["validation"], f.w["validation"],
                                   self.r.seed)
                trees.append(n)
            else:
                v = W.screen(f, self.r.feats, cols)
            vals.append(float(v))
        self.log.add({"judge": self.judge, "groups": key, "n_features": len(cols),
                      "ms": W.subset_cost(key.split(W.SEP), self.costs), "val": float(np.mean(vals)),
                      "val_by_fold": vals, "trees": trees, "seconds": time.perf_counter() - t})
        self.r.log(f"[{self.judge}] {key}: {100 * np.mean(vals):.2f} % ({len(cols)} numbers, "
                   f"{time.perf_counter() - t:.0f} s)")
        return np.array(vals)

    def note(self, **rec):
        self.history.append(rec)
        self.r.log(f"[{self.judge}] {rec['action']}: {rec['sets']} -> {rec['decision']} "
                   f"({100 * rec['gain']:+.2f} points, better in {rec['folds_better']}/5)")

    def try_add(self, kept: list, cands: tuple, stage: str) -> list:
        trial = kept + list(cands)
        ok, gain, nb = beats(self.score(trial), self.score(kept))
        self.note(stage=stage, action="add", sets=" + ".join(cands), gain=gain, folds_better=nb,
                  decision="kept" if ok else "left out", val=float(self.score(trial if ok else kept).mean()))
        return self.float_back(trial, protect=cands) if ok else kept

    def float_back(self, kept: list, protect: tuple) -> list:
        """Drop earlier sets that no longer earn their place; the one whose removal scores best goes first."""
        while len(kept) > 1:
            cur, drops = self.score(kept), []
            for s in kept:
                if s in protect:
                    continue
                without = [x for x in kept if x != s]
                ok, gain, nb = beats(cur, self.score(without))
                if not ok:
                    drops.append((float(self.score(without).mean()), s, gain, nb))
            if not drops:
                break
            val, s, gain, nb = max(drops)
            kept = [x for x in kept if x != s]
            self.note(stage="floating", action="drop", sets=s, gain=-gain, folds_better=5 - nb,
                      decision="dropped", val=val)
        return kept

    def run(self) -> dict:
        alone = {g: float(self.score((g,)).mean()) for g in self.names}
        order = sorted(alone, key=alone.get, reverse=True)
        self.r.log(f"[{self.judge}] order: " + ", ".join(f"{g} {100 * alone[g]:.1f}" for g in order))
        kept = [order[0]]
        for g in order[1:]:                                           # 1: the walk, with floating
            kept = self.try_add(kept, (g,), "walk")
        for g in order:                                               # 2: every left-out set once more
            if g not in kept:
                kept = self.try_add(kept, (g,), "second pass")
        left = [g for g in order if g not in kept]                    # 3: left-out sets in pairs
        for a, b in itertools.combinations(left, 2):
            if a not in kept and b not in kept:
                kept = self.try_add(kept, (a, b), "pairs")
        final = self.canon(kept)
        out = {"judge": JUDGES[self.judge], "min_gain": MIN_GAIN, "min_folds": MIN_FOLDS, "order": order,
               "alone": alone, "history": self.history, "final": final,
               "final_val": float(self.score(kept).mean()), "final_val_by_fold": list(self.score(kept)),
               "n_features": len(W.group_cols(self.groups, final)),
               "ms": W.subset_cost(final.split(W.SEP), self.costs),
               "evaluations": len([x for x in self.log.rows if x["judge"] == self.judge])}
        (OUT_DIR / f"result_{self.judge}.json").write_text(json.dumps(out, indent=2))
        self.r.log(f"[{self.judge}] final ({out['n_features']} numbers, {out['ms']:.0f} ms): {final}, "
                   f"validation {100 * out['final_val']:.2f} %")
        return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judge", choices=["lr", "gb", "both"], default="both")
    p.add_argument("--no-test", action="store_true", help="only the walk, no test of the final set")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    a = p.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    judges = ["lr", "gb"] if a.judge == "both" else [a.judge]
    with keep_awake(), threadpool_limits(W.THREADS):
        table, folds, feats, eng_cols = W.load_all(a.data_dir)
        r = W.Runner(table, folds, feats, OUT_DIR)
        groups = W.cost_groups(eng_cols)
        costs = json.loads((W.OUT_DIR / "costs.json").read_text())   # lesson 23's timing, same laptop
        log = W.Log(OUT_DIR / "steps.jsonl")
        finals = {j: Walk(r, groups, j, log, costs).run()["final"] for j in judges}
        if not a.no_test:
            test_log = W.Log(OUT_DIR / "test.jsonl")
            for j, final in finals.items():
                for model in W.MODELS:
                    for k in range(1, S.N_FOLDS + 1):
                        r.run("test", test_log, W.MAIN, f"walk2 {j}: {final}", W.group_cols(groups, final), model, k,
                              judge=JUDGES[j], ms=W.subset_cost(final.split(W.SEP), costs))
            W.collect(OUT_DIR, "test")
        r.log("finished")


if __name__ == "__main__":
    main()
