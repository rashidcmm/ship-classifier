# %% [markdown]
# # Lesson 04 - A CNN from scratch
#
# A **convolutional neural network (CNN)** slides small learned filters over an image to detect patterns,
# then combines simple patterns into complex ones layer by layer. Here you will:
#
# 1. apply a convolution filter by hand to a spectrogram,
# 2. see what ReLU and pooling do,
# 3. build the small CNN in `src/models.py` (`TinyCNN`),
# 4. write a training loop yourself and watch it learn,
# 5. watch it **overfit**, the most important failure to recognise.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as Fn
import data as D
from device import pick_device
from labels import CLASSES
from models import ShipNet, TinyCNN
from paths import DATA_DIR

torch.manual_seed(0)
mel, table = D.open_features(DATA_DIR)
dev = pick_device()
print("device:", dev)

# %% [markdown]
# ## 1. A convolution by hand
#
# A 3x3 filter is 9 numbers. At every position it multiplies the 9 pixels under it by its 9 numbers and
# adds them up. The filter below responds to **horizontal lines**: a frequency that stays on over time,
# which is exactly what a ship's engine tonal looks like in a spectrogram.

# %%
row = int(table.index[(table.label == "cargo") & (table.mean_distance < 500)][0])
x = torch.from_numpy(mel[row].astype(np.float32))
x = ((x - x.mean()) / x.std())[None, None]                       # shape (batch=1, channels=1, 128, 501)
horizontal = torch.tensor([[-1., -1, -1], [2, 2, 2], [-1, -1, -1]])[None, None]
vertical = horizontal.transpose(2, 3)                              # responds to short clicks / bangs
fig, ax = plt.subplots(1, 3, figsize=(14, 3))
for a, img, title in zip(ax, (x, Fn.conv2d(x, horizontal, padding=1), Fn.conv2d(x, vertical, padding=1)),
                         ("input spectrogram", "horizontal-line filter (tonals)", "vertical-line filter (transients)")):
    a.imshow(img[0, 0], origin="lower", aspect="auto", cmap="magma"); a.set_title(title)
plt.show()

# %% [markdown]
# In a CNN these 9 numbers aren't chosen by hand; they're **learned**, and there are many filters per layer.
#
# ## 2. ReLU and pooling
#
# - **ReLU** keeps positive responses and sets negatives to 0 ("pattern found here" or not).
# - **Max pooling** keeps the strongest response in each 2x2 block. The picture halves in size, so the next
#   layer's 3x3 filter "sees" a larger area. Deeper layers therefore detect bigger, more complex patterns.

# %%
y = Fn.relu(Fn.conv2d(x, horizontal, padding=1))
for _ in range(4):
    print("shape:", tuple(y.shape[2:]))
    y = Fn.max_pool2d(y, 2)
print("shape:", tuple(y.shape[2:]))

# %% [markdown]
# ## 3. TinyCNN
#
# Four blocks of [convolution, batch normalisation, ReLU, max pool] with 16, 32, 64 and 128 filters, then
# **global average pooling** (one number per filter, whatever the clip length), then a linear layer giving
# one score (a **logit**) per class. *Batch normalisation* rescales each layer's outputs to a stable range,
# which makes training faster and more reliable.

# %%
net = ShipNet("tiny_cnn", n_classes=len(CLASSES), pretrained=False).to(dev)
print(net.backbone)
print("trainable numbers (parameters):", sum(p.numel() for p in net.parameters()))

# %% [markdown]
# ## 4. A training loop, written out
#
# One **epoch** = one pass over the training clips, in **batches**. For each batch:
#
# 1. **forward**: the model produces logits,
# 2. **loss**: cross-entropy measures how wrong they are (low when the true class gets a high score),
# 3. **backward**: PyTorch computes how each parameter should change to reduce the loss (the gradient),
# 4. **step**: the optimiser (Adam) nudges every parameter a little in that direction.
#
# After each epoch we measure the **validation** loss and accuracy on clips the model never trains on.

# %%
def subset(split, n):
    """n clips with (as far as possible) the same number per class. Fold 1's validation split is 38 % dredger,
    so a plain random sample would make 'accuracy' mostly mean 'did it say dredger?'."""
    rows = D.select_rows(table, fold=1, split=split, task="class")
    rng, labels, picked = np.random.default_rng(0), table.label.to_numpy()[rows], []
    for c in CLASSES:
        pool = rows[labels == c]
        picked.extend(rng.choice(pool, min(n // len(CLASSES), len(pool)), replace=False))
    return np.sort(picked)

def loaders(n_train, n_val, augment):
    tr, va = subset("train", n_train), subset("validation", n_val)
    norm = D.compute_norm(mel, tr, max_clips=500)
    make = lambda rows, aug, shuffle: torch.utils.data.DataLoader(
        D.MelDataset(DATA_DIR, table, rows, CLASSES, norm, augment=aug), batch_size=32, shuffle=shuffle)
    return make(tr, augment, True), make(va, False, False)

def run_epoch(model, loader, opt=None):
    model.train(opt is not None)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(opt is not None):
        for b in loader:
            xb, yb = b["x"].to(dev), b["y_class"].to(dev)
            logits = model(xb)["logits"]
            loss = Fn.cross_entropy(logits, yb)
            if opt is not None:
                opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(yb); correct += (logits.argmax(1) == yb).sum().item(); total += len(yb)
    return loss_sum / total, correct / total

def train_and_plot(n_train, n_val, epochs, augment, title, lr=1e-3):
    tr, va = loaders(n_train, n_val, augment)
    model = ShipNet("tiny_cnn", n_classes=len(CLASSES), pretrained=False).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        hist.append(run_epoch(model, tr, opt) + run_epoch(model, va))
        if ep == 1 or ep % max(1, epochs // 10) == 0:
            print(f"epoch {ep}: train loss {hist[-1][0]:.3f} acc {hist[-1][1]:.3f} | "
                  f"val loss {hist[-1][2]:.3f} acc {hist[-1][3]:.3f}  ({time.time() - t0:.0f} s)")
    h = np.array(hist)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot(h[:, 0], "o-", label="train"); ax[0].plot(h[:, 2], "o-", label="validation"); ax[0].set_title("loss")
    ax[1].plot(h[:, 1], "o-", label="train"); ax[1].plot(h[:, 3], "o-", label="validation"); ax[1].set_title("accuracy")
    for a in ax: a.set_xlabel("epoch"); a.legend()
    fig.suptitle(title); plt.show()

train_and_plot(2004, 504, epochs=3, augment=True, title="TinyCNN, ~2,000 training clips")

# %% [markdown]
# Chance level is 1/12, about 8 % (the validation clips are balanced across classes). After only 3 short epochs
# the network is already learning something.
#
# ## 5. Overfitting, live
#
# Now train on just **48** clips (4 per class) for many epochs, without augmentation and with a higher learning
# rate. The network starts memorising the 48 clips: training loss keeps falling and training accuracy climbs
# (to roughly 75 % here, towards 100 % with more epochs), while validation accuracy stays around 20 % and validation
# loss stops improving or rises. It learned *those clips* rather than *ship sounds*. When the two curves split apart
# like this, the model is **overfitting**.

# %%
train_and_plot(48, 300, epochs=60, augment=False, lr=3e-3, title="48 clips, no augmentation: overfitting")

# %% [markdown]
# The cures, all used by `src/train.py` (lesson 05):
#
# - **more data**: all ~34,000 training clips,
# - **augmentation**: SpecAugment hides random bands and moments, so memorising exact pixels doesn't help,
# - **dropout**: randomly drops features during training,
# - **early stopping**: keep the epoch with the best validation score,
# - **transfer learning**: start from a network pre-trained on millions of images (EfficientNet-B0).
#
# ## Try it
#
# 1. Rerun the overfitting cell with `augment=True`. Do the curves stay closer together?
# 2. Change the learning rate to `1e-2` and then `1e-4`. What happens to the loss curve?
# 3. Swap the `horizontal` filter for your own 3x3 numbers in section 1. Can you make one that highlights
#    diagonal lines (a tone whose pitch changes, like a passing ship's Doppler shift)?
