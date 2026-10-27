# %% [markdown]
# # Lesson 05 - Training the real model
#
# Lesson 04 wrote a training loop by hand. `src/train.py` is the same loop with everything a real project needs.
# This lesson explains each part, then shows how to read a finished run.
#
# Run it from a terminal (it takes hours, and a notebook stops if its window closes):
# ```
# python src/train.py --task class
# ```

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import data as D
from labels import CLASSES, load_config
from paths import DATA_DIR, RUNS_DIR
from train import lr_lambda_factory

cfg = load_config()
mel, table = D.open_features(DATA_DIR)
RUN = RUNS_DIR / "class_f1_efficientnet_b0_vessel"

# %% [markdown]
# ## 1. Transfer learning: don't start from zero
#
# **EfficientNet-B0** was trained on 1.2 million photos (ImageNet). Its early layers already detect edges,
# lines and textures, and those patterns exist in spectrograms too (a tonal is a horizontal line). We start from
# those weights and **fine-tune** them on ship sounds, which needs far less data and time than starting
# from random numbers.
#
# Photos have 3 colour channels, a spectrogram has 1. `timm` adapts the first layer by summing its 3 colour
# filters into one (`in_chans=1`).
#
# ## 2. The loss, with class weights
#
# Cross-entropy, as in lesson 04, but each class is weighted by `1 / (its share of the training clips)`, so
# rarer classes count as much as common ones.

# %%
rows = D.select_rows(table, fold=1, split="train", task="class")
counts = table.label.iloc[rows].value_counts().reindex(CLASSES)
weights = counts.sum() / (len(CLASSES) * counts)
pd.DataFrame({"training clips": counts, "weight": weights.round(2)})

# %% [markdown]
# Classes are nearly balanced in clips (about 5 hours each), so the weights stay close to 1.
#
# ## 3. Optimiser and learning-rate schedule
#
# - **AdamW** adapts the step size per parameter. *Weight decay* (0.01) gently pulls weights towards zero,
#   which discourages memorising.
# - The **learning rate** (3e-4) is the most important setting. It **warms up** over the first epoch
#   (big early steps can wreck the pretrained weights), then follows a **cosine** curve down to 0, so the last
#   epochs make small, careful adjustments.

# %%
steps_per_epoch, epochs = 100, cfg["epochs"]["class"]
f = lr_lambda_factory(cfg["warmup_epochs"] * steps_per_epoch, epochs * steps_per_epoch)
s = np.arange(epochs * steps_per_epoch)
plt.figure(figsize=(8, 3)); plt.plot(s / steps_per_epoch, [cfg["lr"] * f(i) for i in s])
plt.xlabel("epoch"); plt.ylabel("learning rate"); plt.title("warm-up + cosine schedule"); plt.show()

# %% [markdown]
# ## 4. Augmentation: SpecAugment
#
# Every time a training clip is used, random frequency bands and time spans are hidden (set to the average).
# The model can't rely on one exact detail, so it learns more robust cues.

# %%
r = int(rows[0])
x = torch.from_numpy(mel[r].astype(np.float32))
x = ((x - x.mean()) / x.std())[None]
fig, ax = plt.subplots(1, 3, figsize=(14, 3))
ax[0].imshow(x[0], origin="lower", aspect="auto", cmap="magma"); ax[0].set_title("original")
for i in (1, 2):
    ax[i].imshow(D.spec_augment(x.clone(), cfg["specaugment"])[0], origin="lower", aspect="auto", cmap="magma")
    ax[i].set_title(f"augmented, example {i}")
plt.show()

# %% [markdown]
# ## 5. The rest of `train.py`
#
# | feature | what it does | why |
# |---|---|---|
# | **batch size** | clips per step, chosen from GPU memory (`--batch-size` to override) | bigger = faster, until memory runs out |
# | **mixed precision** | uses 16-bit numbers on NVIDIA GPUs | ~2x faster, half the memory, same accuracy |
# | **validation each epoch** | macro-F1 on the validation clips | the honest progress signal |
# | **best.pt** | saved whenever validation improves | the model you keep |
# | **early stopping** | stops after 6 epochs without improvement | avoids wasting time and overfitting |
# | **last.pt** every epoch + `--resume` | continue after Ctrl+C, a crash or a power cut | long runs become safe |
# | **--max-hours** | stop cleanly after a time limit | train in sessions |
# | **temperature guard** | pauses if the GPU passes 85 °C | protects the laptop |
#
# ## 6. Reading a finished run
#
# A run folder holds `config.json` (every setting used), `log.csv` (one line per epoch), the checkpoints,
# and `done.json` when finished.

# %%
if not (RUN / "log.csv").exists():
    print(f"no run yet: {RUN.name}\n  run:  python src/train.py --task class")
else:
    cfg_run = json.loads((RUN / "config.json").read_text())
    print({k: cfg_run[k] for k in ("backbone", "epochs", "lr", "batch_size", "device", "n_train", "n_val")})
    log = pd.read_csv(RUN / "log.csv")
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
    ax[0].plot(log.epoch, log.train_loss, "o-", label="train"); ax[0].plot(log.epoch, log.val_loss, "o-", label="validation")
    ax[0].set_title("loss"); ax[0].legend()
    ax[1].plot(log.epoch, log.val_metric, "o-"); ax[1].set_title("validation macro-F1")
    ax[1].axvline(log.epoch[log.val_metric.idxmax()], ls="--", c="g", label="best.pt"); ax[1].legend()
    ax[2].plot(log.epoch, log.lr, "o-"); ax[2].set_title("learning rate")
    for a in ax: a.set_xlabel("epoch")
    plt.show()
    print(f"best validation macro-F1 {log.val_metric.max():.3f} at epoch {log.epoch[log.val_metric.idxmax()]}, "
          f"{log.seconds.mean() / 60:.1f} min/epoch, {log.clips_per_s.mean():.0f} clips/s")
    if (RUN / "done.json").exists():
        print("finished:", json.loads((RUN / "done.json").read_text())["reason"])
    else:
        print("not finished yet - continue with:  python src/train.py --task class --resume")

# %% [markdown]
# **How to read the curves:**
# - both losses falling: still learning,
# - training loss falling while validation loss rises: overfitting (early stopping will end the run),
# - a jumpy validation score is normal with ~4,000 validation clips; trust the trend.
#
# ## Try it
#
# 1. After the first run, evaluate it: `python src/evaluate.py runs/class_f1_efficientnet_b0_vessel --mc 20`.
# 2. Look at `runs/class_f1_efficientnet_b0_vessel/eval/confusion.png`. Which classes are confused, and does
#    that make physical sense (e.g. cargo vs tanker, both large ships with slow diesel engines)?
