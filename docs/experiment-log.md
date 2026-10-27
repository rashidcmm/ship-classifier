# Experiment log: what we trained, what we found, what we tried

Fold 1 of the UniqueShip 12-class / 5-hour subset. All numbers below come from the files in `runs/`
(`log.csv`, `done.json`, `eval/metrics.json`). Last updated: 27 October 2026, 23:55.

**Status:** E1, E2 and E5 are finished and evaluated, as is the regularisation trial on validation. Distance,
speed, the three unknown-ship groups, the noise test, the hydrophone shift and the MobileNetV3 trial are running
overnight (section 7).

---

## 1. Setup

| | |
|---|---|
| Data | 43,208 clips of 5 s from 4,218 ships, 2 hydrophones, 12 classes (11 ship types + background) |
| Split | fold 1 of the dataset's **vessel-disjoint** split: no ship appears in more than one of train (34,286 clips), validation (4,445) and test (4,477) |
| Input | log-mel spectrogram, 128 mel bands × 501 frames (10 Hz–16 kHz, 10 ms hop), normalised per band with training-set statistics |
| Network | EfficientNet-B0 pretrained on ImageNet, adapted to 1 input channel; three output heads (class, distance, speed) |
| Training | AdamW, learning rate 3e-4 with 1 warm-up epoch then cosine decay, weight decay 0.01, batch 32, mixed precision, dropout 0.3, SpecAugment (2 frequency masks ≤ 16 bands, 2 time masks ≤ 50 frames), class weights inversely proportional to class frequency |
| Model selection | best epoch by **validation macro-F1**; early stopping after 6 epochs without improvement (maximum 25) |
| Test set | used once per final model, never to choose settings |
| Calibration | temperature scaling fitted on validation, every class given equal weight |
| Hardware | laptop, NVIDIA RTX 3050 Laptop GPU (4 GB), 16 GB RAM; about 190–270 clips/s, with pauses whenever the GPU reached 85 °C |

---

## 2. Results so far (test set, fold 1)

| Run | What it is | Accuracy | Macro-F1 | Notes |
|---|---|---|---|---|
| E1 | classical baseline: per-band mean/std (257 numbers per clip) + gradient boosting | 31.8 % | 0.277 | 2 min on CPU |
| **E2** | **EfficientNet-B0, vessel-disjoint** | **32.4 %** | **0.290** | 11 epochs, 45 min |
| E5 | same network, **leaky split** (clips shuffled without regard to ship) | 85.8 % | 0.857 | 25 epochs, 91 min |
| reg | E2 with stronger regularisation | not tested | val 0.410 (E2: val 0.425) | judged on validation only |

Chance level for 12 classes is 8.3 %. Dredger has no clips in fold 1's test set, so E1 and E2 macro-F1 average
11 classes. The leaky test set contains dredger, so E5 averages 12.

Distance and speed: the baseline's average errors are **274 m** (guessing the median: 403 m) and **2.96 knots**
(median guess: 3.64). The CNN results are pending.

---

## 3. What we observed

### 3.1 On ships it has never heard, the CNN barely beats the simple baseline

Recall per class (test set):

| Class | E1 baseline | E2 CNN | E5 leaky |
|---|---|---|---|
| background | 0.76 | 0.62 | 0.87 |
| cargo | 0.53 | 0.59 | 0.71 |
| dredger | – | – | 0.96 |
| fishing | 0.18 | 0.23 | 0.84 |
| other | 0.13 | 0.10 | 0.86 |
| passengership | 0.23 | 0.29 | 0.90 |
| pilotvessel | 0.00 | 0.00 | 1.00 |
| pleasurecraft | 0.35 | 0.31 | 0.79 |
| rescue | 0.03 | 0.02 | 0.93 |
| sailing | 0.17 | 0.29 | 0.89 |
| tanker | 0.48 | 0.43 | 0.84 |
| tug | 0.36 | 0.39 | 0.72 |

The CNN gains a few points on cargo, fishing, passenger ships and sailing, and loses a few on background and
tanker. Rare classes fail in both: pilot vessel has 2 ships in the whole dataset (1 in this test set) and rescue
has 23. "Other" is a mixed category with no common sound.

### 3.2 The model memorises the training ships after epoch 2

E2 training log (validation set):

| Epoch | Train loss | Val loss | Val macro-F1 |
|---|---|---|---|
| 1 | 1.81 | **1.65** | 0.321 |
| 2 | 1.29 | 1.67 | 0.395 |
| 3 | 1.05 | 2.14 | 0.365 |
| 4 | 0.87 | 1.91 | 0.412 |
| **5** | 0.72 | 1.91 | **0.425** (best: kept as `best.pt`) |
| 8 | 0.44 | 2.73 | 0.376 |
| 11 | 0.27 | 2.70 | 0.403 (early stop) |

Training loss falls steadily while validation loss rises from epoch 2. This is the classic picture of
overfitting: the network keeps getting better on the training ships and worse on new ones.

### 3.3 Validation is a poor guide on this fold

Validation macro-F1 was 0.425, but test macro-F1 was only 0.290. Two reasons:

- **The splits contain different classes.** Validation is 38 % dredger clips, from very few ships, and dredger
  doesn't appear in test at all.
- **Some classes have very few ships**, so one ship more or less changes a class's score a lot.

Because of this, small differences between settings (for example 0.410 against 0.425) are within noise
(section 6).

### 3.4 Recognition gets worse with distance

| Distance to hydrophone | Test clips | E2 accuracy | E5 (leaky) accuracy |
|---|---|---|---|
| < 700 m | 770 | 43 % | 81 % |
| 700–1,400 m | 1,532 | 33 % | 89 % |
| > 1,400 m | 1,804 | 22 % | 85 % |

For new ships, accuracy halves from near to far, as the ship's sound fades into background noise. With the
leaky split there's no such drop: a ship the model has heard before is recognised even from far away.

### 3.5 The model is overconfident; temperature scaling fixes the average

| E2 | Before | After temperature scaling (T = 2.18) |
|---|---|---|
| ECE (confidence error) | 0.330 | **0.093** |
| Log-loss (NLL) | 2.63 | 1.94 |
| Brier score | 0.94 | 0.80 |

T > 1 means the raw model claims much more certainty than it has. Monte Carlo dropout (20 passes) left both
accuracy (32.5 %) and ECE (0.325) unchanged, so temperature scaling is the method that works here.

### 3.6 Combining the clips of one ship helps

In use, one passing ship gives several 5-s clips. Averaging the predicted probabilities over all test clips of the
same ship (MMSI):

| | Accuracy |
|---|---|
| per clip, ship clips only | 29.8 % |
| per ship, 217 ships with ≥ 3 clips: one clip at a time | 30.3 % |
| per ship, 217 ships with ≥ 3 clips: clips averaged | **37.8 %** |

Tug improves from 39 % to 62 % at ship level, sailing from 29 % to 38 %. (This analysis was a one-off script.
`predict.py` combines the windows of a longer recording by majority vote of the per-window classes instead;
averaging probabilities as here has not been compared with that.)

### 3.7 The diagnosis: the model learns ship identity, not ship type (E5)

The leaky split (E5) uses the same clip counts per split, but shuffles clips without regard to which ship they
come from. The test ships were therefore also heard during training.

| | Vessel-disjoint (E2) | Leaky (E5) |
|---|---|---|
| Test accuracy | 32.4 % | **85.8 %** |
| Test macro-F1 | 0.290 | **0.857** |
| Validation loss over training | rises from epoch 2 | falls steadily (1.37 → 0.58) |
| Rare classes (rescue, pilot vessel) | 2 % and 0 % | 93 % and 100 % |
| ECE after temperature scaling | 0.093 | 0.016 |

Same network, same settings, same number of clips: **53 points of accuracy** come only from letting the model hear
the test ships before. So:

1. The network can tell individual ships apart very well. Engines, propellers and hulls give each ship a
   recognisable sound.
2. With only a few ships per class, it doesn't learn what makes a *tanker* sound like a tanker in general. The
   sound features that identify one training ship don't carry over to other ships of the same class.
3. A random (leaky) split, as used in some earlier work, would report 86 % for this model. The honest number for
   new ships is 32 %. This is the inflation our presentation warned about.

---

## 4. What we tried to improve the model

### 4.1 Stronger regularisation (`configs/strong_regularization.yaml`)

| Setting | E2 | Trial |
|---|---|---|
| Learning rate | 3e-4 | 1e-4 |
| Weight decay | 0.01 | 0.05 |
| Dropout | 0.3 | 0.5 |
| SpecAugment | 2 × 16 bands, 2 × 50 frames | 3 × 24 bands, 3 × 80 frames |

| Epoch | Train loss | Val loss | Val macro-F1 |
|---|---|---|---|
| 1 | 2.11 | **1.59** | 0.310 |
| **4** | 1.23 | 1.69 | **0.410** (best) |
| 9 | 0.77 | 2.25 | 0.356 |
| 10 | 0.72 | 2.40 | 0.369 (early stop) |

**Result: no improvement** (validation 0.410 against 0.425; the test set was not used). Memorisation became slower
(training loss 0.77 at epoch 9 against 0.38 for E2), but validation loss still rose after epoch 4. This agrees
with 3.7: the problem isn't too little regularisation, but too few different ships per class to learn general
class sounds from.

### 4.2 Smaller network (MobileNetV3-Large)

The idea: less capacity means less room to memorise individual ships. The first attempt crashed at the start
because of a bug in our model code. MobileNetV3 outputs 1,280 features, but the output layer was sized from timm's
`num_features` (960). The bug was fixed (`src/models.py`), with a test that now builds every backbone. The rerun is
scheduled overnight; the result will be judged on validation only.

### 4.3 Shorter runs for the review deadline (`configs/fast.yaml`)

Early stopping after 3 epochs without improvement instead of 6, and at most 12 epochs. E2's best epoch was 5, so
this mainly removes epochs that didn't help. All overnight runs use it.

---

## 5. Engineering notes from the training runs

- **Heat:** the GPU reached 86–87 °C every epoch. The temperature guard paused training until it cooled below
  75 °C, costing about 45–105 s per 4-minute epoch.
- **Memory:** the first E2 evaluation was stopped when the laptop ran low on RAM, while other programs (browser,
  IDE, chat apps) were also open. Training was already saved, so only the evaluation was rerun. Long runs now start in a separate window.
- **Resuming:** every epoch is saved (`last.pt`), so a crash costs at most one epoch.

---

## 6. Limitations

- **One fold, one random seed.** Differences of a few hundredths in macro-F1 are not meaningful. The
  baseline is being run on folds 2–5 tonight. E2 on all 5 folds (about 3.5 h) was left out because of the review
  deadline.
- **Few ships in rare classes:** pilot vessel 2, dredger 6 and rescue 23 in the whole dataset. Their scores rest on
  one or a handful of ships.
- **Validation and test differ in class mix** (section 3.3).
- Only two hydrophones at one location, one ship at a time, ships up to about 2 km away.

---

## 7. Pending (running overnight, 27–28 October)

| Priority | Run | Purpose |
|---|---|---|
| 1 | distance (E9), speed (E10), initialised from E2 | the project's distance and speed objectives |
| 1 | unknown ships, group A (E3) + MC dropout | detecting ship types never seen in training |
| 1 | noise test on E2 (E8): 20, 10, 0 dB | robustness |
| 2 | baseline, folds 2–5 | 5-fold average for the baseline |
| 2 | unknown ships, groups B and C | average unknown detection over 3 groups |
| 2 | hydrophone shift (E7) | train on one microphone, test on the other |
| 3 | MobileNetV3 | tuning comparison (validation only) |

Left out for time: the 3-model deep ensemble, and E2 on folds 2–5.

---

## 8. Ideas not yet tried

These follow from 3.7. None has been tested.

- **Ship-level prediction:** report a decision per ship passage (averaging clips, 3.6), as in real use.
- **Augmentation that changes ship identity but keeps the class:** mixup between ships of the same class, pitch or
  speed shifts, mixing in recorded background noise.
- **Class-balanced sampling by ship,** so that ships with many clips don't dominate training.
- **Grouping rare classes** (pilot vessel, rescue, dredger) or reporting them separately, since 2–23 ships can't
  support a reliable class.
- **5-fold evaluation** of the final model, to report a mean ± standard deviation.

---

## 9. Reproduce

```
python src/baseline_classical.py --fold 1                                      # E1
python src/train.py --task class                                               # E2
python src/evaluate.py runs/class_f1_efficientnet_b0_vessel --mc 20
python src/train.py --task class --config configs/strong_regularization.yaml --name class_f1_efficientnet_b0_vessel_reg
python src/train.py --task class --split-mode leaky                            # E5
python src/evaluate.py runs/class_f1_efficientnet_b0_leaky
```
