"""Run every experiment of README section 4, one after another, and pick up where it stopped.

  python src/run_queue.py               # start, or continue after a shutdown, crash or Ctrl+C
  python src/run_queue.py --dry-run     # list what is still to do
  python src/run_queue.py --workers 2   # fewer data-loading processes if RAM is short

A finished training run (done.json) is skipped, a half-finished one continues from its last.pt.
Other finished steps (baseline, evaluations) are listed in runs/queue_done.txt; delete a line to redo one.
The queue stops at the first step that fails.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from keepawake import keep_awake
from paths import ROOT, RUNS_DIR

E2 = "class_f1_efficientnet_b0_vessel"
DIST, SPEED = "distance_f1_efficientnet_b0_vessel", "speed_f1_efficientnet_b0_vessel"
LEAKY, HYD = "class_f1_efficientnet_b0_leaky", f"{E2}_hyd"
DONE_FILE = "queue_done.txt"


def plan(runs, workers):
    """All steps in README order: (step id, script arguments). A training step's id is 'train <run name>'."""
    w = ["--workers", str(workers)]

    def train(name, *args):
        return f"train {name}", ["src/train.py", *args, "--name", name, "--runs-dir", str(runs), *w]

    def evaluate(name, *args, tag="eval"):
        return f"{tag} {name}", ["src/evaluate.py", str(runs / name), *args, *w]

    mc = ["--mc", "20"]
    return [
        ("E1 baseline", ["src/baseline_classical.py", "--fold", "1", "--runs-dir", str(runs)]),
        train(E2, "--task", "class"),
        evaluate(E2, *mc),
        *(train(f"{E2}_h{g}", "--task", "class", "--holdout", g) for g in "ABC"),
        *(train(f"{E2}_hA_s{s}", "--task", "class", "--holdout", "A", "--seed", str(s)) for s in (1, 2)),
        evaluate(f"{E2}_hA", *mc, "--ensemble", str(runs / f"{E2}_hA_s1"), str(runs / f"{E2}_hA_s2")),
        evaluate(f"{E2}_hB", *mc),
        evaluate(f"{E2}_hC", *mc),
        train(DIST, "--task", "distance", "--init-from", str(runs / E2 / "best.pt")),
        evaluate(DIST),
        train(SPEED, "--task", "speed", "--init-from", str(runs / E2 / "best.pt")),
        evaluate(SPEED),
        train(LEAKY, "--task", "class", "--split-mode", "leaky"),
        evaluate(LEAKY),
        train(HYD, "--task", "class", "--hydrophone-shift"),
        evaluate(HYD),
        evaluate(E2, "--noise", "20", "10", "0", tag="noise"),
    ]


def done_steps(runs):
    p = Path(runs) / DONE_FILE
    return set(p.read_text().splitlines()) if p.exists() else set()


def mark_done(runs, step):
    Path(runs).mkdir(parents=True, exist_ok=True)
    with open(Path(runs) / DONE_FILE, "a") as f:
        f.write(step + "\n")


def pending_steps(runs, workers=4):
    """Steps still to do; a training run that has a last.pt gets --resume."""
    runs = Path(runs)
    finished = done_steps(runs)
    todo = []
    for step, cmd in plan(runs, workers):
        if step in finished:
            continue
        if step.startswith("train "):
            run_dir = runs / step.split(" ", 1)[1]
            if (run_dir / "done.json").exists():
                continue
            if (run_dir / "last.pt").exists():
                cmd = [*cmd, "--resume"]
        todo.append((step, cmd))
    return todo


def run_step(cmd):
    return subprocess.run([sys.executable, *cmd], cwd=ROOT).returncode


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true", help="only list the steps still to do")
    a = p.parse_args(argv)

    todo = pending_steps(a.runs_dir, a.workers)
    print(f"{len(todo)} steps to do" + ("" if todo else " - everything is finished"), flush=True)
    if a.dry_run:
        for step, cmd in todo:
            print(f"  {step:<45} python {' '.join(cmd)}")
        return 0
    with keep_awake():
        for i, (step, cmd) in enumerate(todo, 1):
            print(f"\n=== [{i}/{len(todo)}] {step}  ({time.strftime('%Y-%m-%d %H:%M')}) ===", flush=True)
            start = time.time()
            code = run_step(cmd)
            if code != 0:
                print(f"=== {step} FAILED (exit code {code}). Fix the problem, then run this script again: "
                      f"finished steps are skipped and a half-trained run continues. ===", flush=True)
                return 1
            mark_done(a.runs_dir, step)
            print(f"=== {step} done in {(time.time() - start) / 60:.0f} min ===", flush=True)
    print("\nall steps finished - see notebooks/10_stress_tests_and_report.ipynb for the report", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
