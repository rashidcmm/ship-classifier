# %% [markdown]
# # Lesson 08 - Confidence you can trust
#
# When the model says "tanker, 90 %", it should be right about 90 % of the time. Deep networks are often
# **overconfident**. This lesson measures that (**calibration**) and fixes it (**temperature scaling**), then
# compares two ways to estimate uncertainty: **Monte Carlo dropout** and **deep ensembles** (presentation slide 4).

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import metrics as M
import reliability as R
from paths import RUNS_DIR

RUN = RUNS_DIR / "class_f1_efficientnet_b0_vessel"
EVAL = RUN / "eval"

# %% [markdown]
# ## 1. Reliability diagrams and ECE
#
# Group test clips by the model's confidence (0-10 %, 10-20 %, ...) and compare with the accuracy actually achieved
# in each group. A perfectly calibrated model sits on the diagonal.
#
# **ECE** (expected calibration error) = the average gap between confidence and accuracy, weighted by how many
# clips fall in each group. 0 is perfect.
#
# Here is a synthetic, overconfident model to show the idea:

# %%
rng = np.random.default_rng(0)
z = rng.normal(0, 1.5, size=(5000, 5))
y = np.array([rng.choice(5, p=p) for p in R.softmax(z)])       # labels consistent with the logits: calibrated
for name, logits in (("calibrated", z), ("overconfident (logits x 3)", 3 * z)):
    print(f"{name:<28} ECE = {M.ece(R.softmax(logits), y):.3f}")

# %% [markdown]
# ## 2. Temperature scaling
#
# Divide all logits by one number **T** before the softmax. T > 1 softens overconfident probabilities, T < 1
# sharpens them. T is chosen to minimise the log-loss on the **validation** set. It never changes which class
# wins, only how sure the model claims to be.

# %%
T = R.fit_temperature(3 * z, y)
print(f"fitted T = {T:.2f} (the true answer is 3)")
print(f"ECE after dividing by T: {M.ece(R.softmax(3 * z, T), y):.3f}")

# %% [markdown]
# ## 3. On your model
#
# `evaluate.py` fits T on validation, saves it in `eval/calibration.json`, and reports ECE, log-loss (NLL) and
# Brier score before and after. Every class gets equal total weight in the fit, because fold 1's validation split
# is 38 % dredger (lesson 01) and would otherwise decide T almost by itself.

# %%
if not (EVAL / "metrics.json").exists():
    print(f"no evaluation yet:\n  python src/train.py --task class\n  python src/evaluate.py {RUN.relative_to(ROOT).as_posix()} --mc 20")
else:
    m = json.loads((EVAL / "metrics.json").read_text())
    c = m["calibration"]
    display(pd.DataFrame({"before": [c["ece_before"], c["nll_before"], c["brier_before"]],
                          "after temperature": [c["ece_after"], c["nll_after"], c["brier_after"]]},
                         index=["ECE", "NLL", "Brier"]).round(4))
    print(f"temperature T = {c['temperature']:.2f}  ({'model was overconfident' if c['temperature'] > 1 else 'model was underconfident'})")
    from IPython.display import Image, display as show
    show(Image(filename=str(EVAL / "reliability_before_after.png")))

# %% [markdown]
# ## 4. Monte Carlo dropout vs deep ensembles
#
# Temperature scaling fixes the *average* confidence. To know *which* clips are uncertain, we can ask several
# slightly different models and see whether they agree:
#
# | method | how | cost |
# |---|---|---|
# | **MC dropout** | keep dropout switched on at test time and run the same model 20x; each run drops different features | 1 training run, 20x slower prediction |
# | **Deep ensemble** | train 3 models from different random starts (`--seed 0/1/2`) and average them | 3x training |
#
# From the samples come two scores: **entropy** of the averaged prediction (overall uncertainty) and **mutual
# information** (how much the samples *disagree*: uncertainty from the model itself, which is high for unfamiliar
# inputs).
#
# The ensemble is trained for group A of lesson 07 (its known classes), so one evaluation answers both questions,
# calibration here and unknown detection there:
# ```
# python src/train.py --task class --holdout A --seed 1
# python src/train.py --task class --holdout A --seed 2
# python src/evaluate.py runs/class_f1_efficientnet_b0_vessel_hA --mc 20 \
#     --ensemble runs/class_f1_efficientnet_b0_vessel_hA_s1 runs/class_f1_efficientnet_b0_vessel_hA_s2
# ```
# Accuracy and ECE below are measured on the known classes only.

# %%
found = False
for run in (RUN, RUNS_DIR / "class_f1_efficientnet_b0_vessel_hA"):
    p = run / "eval" / "metrics.json"
    if not p.exists():
        continue
    m = json.loads(p.read_text())
    rows = {"single model": {"accuracy": m["closed_set"]["accuracy"], "ECE": m["calibration"]["ece_before"]}}
    if "mc_dropout" in m:
        rows[f"MC dropout ({m['mc_dropout']['samples']} samples)"] = {"accuracy": m["mc_dropout"]["accuracy"], "ECE": m["mc_dropout"]["ece"]}
    if "ensemble" in m:
        rows[f"ensemble ({len(m['ensemble']['members'])} models)"] = {"accuracy": m["ensemble"]["accuracy"], "ECE": m["ensemble"]["ece"]}
    if len(rows) > 1:
        found = True
        print(run.name); display(pd.DataFrame(rows).T.round(4))
if not found:
    print("no MC dropout / ensemble results yet: evaluate with --mc 20, and train the group-A seeds (see above)")

# %% [markdown]
# For unknown-ship detection with these scores, see the `mc_entropy`, `mc_mi` and `ens_entropy` rows in
# lesson 07's table (evaluate the holdout runs with `--mc 20`).
#
# ## Try it
#
# 1. Is your model over- or underconfident (T above or below 1)?
# 2. Does the ensemble improve accuracy, calibration, or both? Is it worth 3x the training time?
# 3. In `eval/predictions.parquet`, sort test clips by `mc_mi`. Listen to (lesson 01) or plot the 5 most
#    uncertain clips. What do they have in common?
