# %% [markdown]
# # Lesson 03 - A classical baseline, and how to choose an algorithm
#
# Before training a deep network, always ask: **how well does something simple do?** If a simple method
# gets close, the CNN isn't worth its cost. If it's far behind, you've learned how much the CNN adds.
#
# You will:
# 1. turn each spectrogram into a short list of hand-made numbers (**features**),
# 2. compare three algorithms on the **validation** set (that's algorithm selection),
# 3. test only the winner, **once**, on the test set,
# 4. learn why accuracy can mislead and what macro-F1 fixes.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import time
import matplotlib.pyplot as plt
import numpy as np
import data as D
import metrics as M
from baseline_classical import band_stats, features_for
from labels import CLASSES
from paths import DATA_DIR

mel, table = D.open_features(DATA_DIR)
index = {c: i for i, c in enumerate(CLASSES)}
rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. Hand-made features
#
# For each of the 128 mel bands, take the **average** level and how much it **varies** over the 5 seconds,
# plus the overall loudness: 257 numbers per clip instead of 64,128. The algorithm never sees the full picture,
# only this summary.

# %%
example = band_stats(mel[:1])
print("features per clip:", example.shape[1])
plt.figure(figsize=(9, 2.5)); plt.plot(example[0, :128], label="mean per band"); plt.plot(example[0, 128:256], label="std per band")
plt.xlabel("mel band"); plt.legend(); plt.show()

# %% [markdown]
# ## 2. Train / validation / test, in practice
#
# To keep this lesson quick we train on a random 6,000 of the ~34,000 training clips
# (`baseline_classical.py` uses up to 20,000).

# %%
tr = np.sort(rng.choice(D.select_rows(table, fold=1, split="train", task="class"), 6000, replace=False))
va = D.select_rows(table, fold=1, split="validation", task="class")
te = D.select_rows(table, fold=1, split="test", task="class")
t0 = time.time()
Xtr, Xva, Xte = features_for(mel, tr), features_for(mel, va), features_for(mel, te)
ytr, yva, yte = (np.array([index[c] for c in table.label.iloc[r]]) for r in (tr, va, te))
print(f"features ready in {time.time() - t0:.0f} s: train {Xtr.shape}, validation {Xva.shape}, test {Xte.shape}")

# %% [markdown]
# ## 3. Algorithm selection on the validation set
#
# Three classic choices:
#
# | algorithm | idea | strengths |
# |---|---|---|
# | logistic regression | one weighted sum per class | fast, simple, but only straight-line boundaries |
# | random forest | many decision trees vote | handles non-linear patterns, little tuning |
# | gradient boosting | trees added one by one, each fixing the last one's mistakes | usually the strongest on tables of numbers |
#
# We compare them on **validation** macro-F1. The test set stays untouched, because choosing a model by its
# test score would make the test score optimistic (you would have tuned to it).

# %%
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

candidates = {
    "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "random forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=0),
    "gradient boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=0),
}
scores = {}
for name, model in candidates.items():
    t0 = time.time()
    model.fit(Xtr, ytr)
    p = model.predict(Xva)
    scores[name] = M.macro_f1(yva, p, len(CLASSES))
    print(f"{name:<20} validation accuracy {M.accuracy(yva, p):.3f}  macro-F1 {scores[name]:.3f}  ({time.time() - t0:.0f} s)")
best = max(scores, key=scores.get)
print("\nchosen on validation:", best)

# %% [markdown]
# ## 4. The winner, tested once

# %%
p = candidates[best].predict(Xte)
print(f"{best} on TEST: accuracy {M.accuracy(yte, p):.3f}, macro-F1 {M.macro_f1(yte, p, len(CLASSES)):.3f}")
cm = M.confusion(yte, p, len(CLASSES))
with np.errstate(invalid="ignore"):
    plt.figure(figsize=(6, 5)); plt.imshow(np.nan_to_num(cm / cm.sum(1, keepdims=True)), cmap="Blues", vmin=0, vmax=1)
plt.xticks(range(12), CLASSES, rotation=70); plt.yticks(range(12), CLASSES)
plt.xlabel("predicted"); plt.ylabel("true"); plt.title("confusion matrix (row = true class)"); plt.colorbar(); plt.show()

# %% [markdown]
# ## 5. Accuracy vs macro-F1
#
# **Accuracy** = fraction of clips classified correctly. It hides problems with small classes: if a test set
# were 90 % cargo, "always say cargo" would score 90 %.
#
# **F1** for one class combines *precision* (when I say "tug", how often am I right?) and *recall* (of all real
# tugs, how many did I find?). **Macro-F1** averages F1 over classes, so every class counts equally.

# %%
y_toy = np.array([0] * 90 + [1] * 10)
always_zero = np.zeros(100, int)
print("always predicting class 0: accuracy", M.accuracy(y_toy, always_zero), "| macro-F1", round(M.macro_f1(y_toy, always_zero, 2), 3))

# %% [markdown]
# ## Try it
#
# 1. Use 20,000 training clips instead of 6,000. How much does the score improve? (More data usually helps,
#    with diminishing returns.)
# 2. Drop the std features (`Xtr[:, :128]` etc.). Does variation over time carry information?
# 3. Run the full baseline from a terminal: `python src/baseline_classical.py --fold 1`. It also predicts
#    distance and speed. This is experiment **E1**, the number your CNN must beat.
