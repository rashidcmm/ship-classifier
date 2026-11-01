# Experiment log: what we trained, what we found, what we tried

Fold 1 of the UniqueShip 12-class / 5-hour subset, unless a row says otherwise. All numbers below come from the
files in `runs/` (`log.csv`, `done.json`, `eval/metrics.json`, `metrics.json`). Last updated: 1 November 2026.

**Status:** everything planned for the review is finished: E1 (5 folds, basic and rich features, 12 and 9 classes),
E2, the **9-class model E2-9** (decided 28 October, section 4.5), E5, distance (E9), speed (E10), unknown ships
groups A, B and C (E3), the noise test (E8), the hydrophone shift (E7) and the MobileNetV3 trial. Lessons 03–10 were
executed with these results (section 7). On 28 October two **5-class models** were added (E2-5S, and E2-5U on
all 5 folds, section 4.7), and the dataset's own paper was read (section 4.8). Lesson 11
(`notebooks/11_results_and_five_classes.ipynb`) recomputes the main results of this log, with figures, from the saved run files.
On 1 November the project was **simplified to 6 classes with classical models only** (section 4.9, lesson 13):
background, cargo, fishing, pleasure craft, tanker, tug, on new balanced folds, with interpretable sound features.

**In one paragraph:** on ships it has never heard, every method we tried lands at about the same accuracy: 32 % for
12 classes and 37 % for 9 classes (CNN and classical baseline alike), against 86 % when the test ships were heard in
training (3.7). The models learn individual ships, not ship types, because some classes have very few ships.
Temperature scaling makes the confidence trustworthy on average, but unknown ship types and added noise are not
detected (3.9, 3.11). Distance and speed are estimated about as well by the baseline as by the CNN (2.2, 3.10).
With **5 broader classes** the same CNN reaches **63.5 %** (ShipsEar size groups, fold 1) and, on the 5-class task
of the dataset's own paper, **66.8 ± 4.4 %** over 5 folds (balanced accuracy 65.5 ± 2.3 %): level with the paper's
best model (66.5 %), which had 8.5 times more audio per class (4.7, 4.8).

---

## 1. Setup

| | |
|---|---|
| Data | 43,208 clips of 5 s from 4,218 ships, 2 hydrophones, 12 classes (11 ship types + background) |
| Split | fold 1 of the dataset's **vessel-disjoint** split: no ship appears in more than one of train (34,286 clips), validation (4,445) and test (4,477) |
| Input | log-mel spectrogram, 128 mel bands × 501 frames (10 Hz–16 kHz, 10 ms hop), normalised per band with training-set statistics |
| Network | EfficientNet-B0 pretrained on ImageNet, adapted to 1 input channel; three output heads (class, distance, speed) |
| Training | AdamW, learning rate 3e-4 with 1 warm-up epoch then cosine decay, weight decay 0.01, batch 32, mixed precision, dropout 0.3, SpecAugment (2 frequency masks ≤ 16 bands, 2 time masks ≤ 50 frames), class weights inversely proportional to class frequency |
| Model selection | best epoch by **validation macro-F1** (distance and speed: validation error); early stopping after 6 epochs without improvement, maximum 25 (overnight runs: `configs/fast.yaml`, 3 and 12) |
| Test set | used once per final model, never to choose settings |
| Calibration | temperature scaling fitted on validation, every class given equal weight |
| Hardware | laptop, NVIDIA RTX 3050 Laptop GPU (4 GB), 16 GB RAM; about 140–270 clips/s, with pauses whenever the GPU reached 85 °C |

---

## 2. Results so far (test set)

### 2.1 Ship class

| Run | What it is | Accuracy | Macro-F1 | Notes |
|---|---|---|---|---|
| E1 | classical baseline: per-band mean/std (257 numbers per clip) + gradient boosting | 31.8 % | 0.277 | 3 min on CPU |
| E1, 5 folds | the same on folds 1–5 | 28.4 ± 9.1 % | 0.271 ± 0.067 | fold 2 is distorted (3.8); without it 32.4 % / 0.299 |
| E1 rich | E1 + MFCCs, percentiles, MFCC change (761 numbers, section 4.4) | 34.0 % | 0.292 | 6 min on CPU |
| E1 rich, 5 folds | the same on folds 1–5 | 29.5 ± 9.5 % | 0.281 ± 0.068 | section 4.4 |
| E1-9 | E1 on 9 classes (section 4.5) | 34.6 % | 0.349 | 5 folds: 33.6 ± 6.6 % / 0.333 ± 0.050 |
| E1-9 rich | E1 rich on 9 classes | 36.4 % | 0.371 | as good as the 9-class CNN |
| **E2** | **EfficientNet-B0, vessel-disjoint** | **32.4 %** | **0.290** | 11 epochs, 45 min |
| **E2-9** | **E2 with pilot vessel, dredger and rescue trained as "other": 9 classes (section 4.5)** | **36.7 %** | **0.370** | 10 epochs, 42 min |
| **E2-5S** | **E2 trained on 5 size groups as in ShipsEar; "other" left out (section 4.7)** | **63.5 %** | **0.604** | 5 epochs, 19 min |
| **E2-5U** | **E2 trained on the 5 classes of the dataset's paper: background, cargo, passenger ship, tanker, tug (4.7)** | **62.8 %** | **0.632** | 9 epochs, 16 min |
| **E2-5U, 5 folds** | **the same on folds 1–5** | **66.8 ± 4.4 %** | **0.644 ± 0.020** | balanced accuracy 65.5 ± 2.3 % (4.7) |
| E5 | same network, **leaky split** (clips shuffled without regard to ship) | 85.8 % | 0.857 | 25 epochs, 91 min |
| reg | E2 with stronger regularisation | not tested | val 0.410 (E2: val 0.425) | judged on validation only |
| MobileNetV3 | E2's setup with MobileNetV3-Large (section 4.2) | not tested | val 0.381 (E2: val 0.425) | judged on validation only; 25 min |
| E7 | trained on hydrophone 2523 only, tested on 2556 | 40.2 % | 0.304 | smaller test set, 29 % background: see 3.13 |
| E3 A | 9 known classes (fishing, sailing, tanker held out as unknown) | 44.8 % | 0.377 | known classes only; 18 min |
| E3 B | 9 known classes (cargo, pleasure craft, rescue held out) | 44.4 % | 0.385 | known classes only; 16 min |
| E3 C | 9 known classes (tug, passenger ship, other held out) | 42.1 % | 0.363 | known classes only; crashed once and resumed (section 5) |

Chance level is 8.3 % for 12 classes, 11.1 % for 9 and 20 % for 5. Dredger has no clips in fold 1's test set, so E1 and E2
macro-F1 average 11 classes. The leaky test set contains dredger, so E5 averages 12. E3 scores higher partly because
it has fewer classes to choose from.

### 2.2 Distance and speed

| Run | Distance error (mean) | Speed error (mean) |
|---|---|---|
| Always guess the median | 403 m | 3.64 knots |
| E1 classical baseline | 274 m | 2.96 knots |
| E1 rich | 270 m | 2.94 knots |
| **E9 / E10 CNN** (initialised from E2) | **275 m** | **2.94 knots** |

The CNN is no better than the classical baseline for distance or speed. Section 3.10 scores distance as classes.

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
overfitting: the network keeps getting better on the training ships and worse on new ones. The E3 runs show the
same: best epoch 2, validation loss rising afterwards.

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
leaky split there's no such drop: a ship the model has heard before is recognised even from far away. E3 A shows
the same pattern (56 %, 44 % and 32 %).

### 3.5 The model is overconfident; temperature scaling fixes the average

| E2 | Before | After temperature scaling (T = 2.18) |
|---|---|---|
| ECE (confidence error) | 0.330 | **0.093** |
| Log-loss (NLL) | 2.63 | 1.94 |
| Brier score | 0.94 | 0.80 |

T > 1 means the raw model claims much more certainty than it has. Monte Carlo dropout (20 passes) left both
accuracy (32.5 %) and ECE (0.325) unchanged, so temperature scaling is the method that works here. The E3 models
agree: ECE 0.227 → 0.084 (A), 0.172 → 0.043 (B) and 0.276 → 0.115 (C).

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

### 3.8 The folds differ a lot, because a few ships have thousands of clips

The 5-fold baseline (E1) scores 31.8, **12.4**, 31.1, 31.4 and 35.4 % accuracy. Fold 2 is the odd one out, and
the reason is its test set:

| Fold | Test clips | Largest share of the test set |
|---|---|---|
| 1 | 4,477 | tanker 12 % |
| **2** | 5,659 | **one single pilot vessel: 3,565 clips (63 %)** |
| 3 | 3,636 | rescue 13 % |
| 4 | 4,280 | dredger 36 % (from very few ships) |
| 5 | 3,605 | sailing 13 % |

Pilot vessel has only 2 ships in the dataset: in fold 2 the model learns the class from 36 clips of one ship and is
tested on 3,565 clips of the other. Its speed (median 22 knots) also explains fold 2's large speed error
(9.45 knots against about 3 elsewhere). Averages over folds should therefore be read with the per-fold numbers,
and macro-F1 (every class counts equally) is more stable than accuracy here.

### 3.9 Unknown ship types are not detected (E3)

Each E3 model is trained without 3 classes and should flag their test clips as "unknown". AUROC is 0.5 for a
random guess and 1.0 for perfect detection:

| Unknown score | Group A (fishing, sailing, tanker) | Group B (cargo, pleasure craft, rescue) | Group C (tug, passenger ship, other) | Mean |
|---|---|---|---|---|
| Max softmax probability | 0.50 | 0.58 | 0.51 | 0.53 |
| Energy | 0.52 | 0.58 | 0.57 | 0.56 |
| MC dropout entropy | 0.51 | 0.59 | 0.51 | 0.53 |
| MC dropout mutual information | 0.48 | 0.47 | 0.56 | 0.50 |

The model is as confident on ship types it has never heard as on known ones. This fits 3.7: if it can't capture
what a known class sounds like in general, it can't notice when a sound fits no known class either.

### 3.10 Distance as classes: the CNN matches the baseline

A related project (section 4.4) predicts distance as classes instead of a number. To compare, every distance
result is now also scored in 250 m bins (8 bins from 0 to 2 km, the last one open-ended):

| Distance scored as classes (test) | Right bin | Right bin or the next one | Near (< 700 m) / far, balanced |
|---|---|---|---|
| Always guess the median | 19.2 % | 54.1 % | 50 % |
| E1 baseline | 28.2 % | 71.4 % | 72.3 % |
| E1 rich | 29.7 % | 72.7 % | 73.1 % |
| E9 CNN | 29.8 % | 72.7 % | **75.6 %** |

All methods land in the same place. The CNN is slightly better only at telling near from far.

### 3.11 Noise makes the model worse and *more* sure of itself (E8)

Noise from the recordings was added to 1,000 test clips at three levels (SNR, lower = noisier):

| | Clean | 20 dB | 10 dB | 0 dB |
|---|---|---|---|---|
| Accuracy | 34.6 % | 22.5 % | 15.8 % | 13.0 % |
| Clips flagged "unknown" | 5.4 % | 3.5 % | 2.9 % | 0.8 % |

A reliable model would say "not sure" more often as the sound gets noisier. Ours does the opposite.

### 3.12 How the question is asked changes the number

From E2's test predictions (no retraining):

| Measure | Result |
|---|---|
| Accuracy, every clip | 32.4 % |
| True class among the 2 most likely | 53.9 % |
| True class among the 3 most likely | 66.4 % |
| Accuracy on the most confident 50 % of clips | 42.4 % |
| Accuracy on the most confident 20 % of clips | 51.8 % |
| Accuracy on the most confident 10 % of clips | 57.7 % |

Within known ship types, confidence does carry information: the more confident the model is, the more often it's
right. A system that answers only when confident (and says "not sure" otherwise) is right more often, but answers
fewer clips. Each of these numbers must be reported with its definition; none of them is "the accuracy".

### 3.13 A different hydrophone costs about 7 points (E7)

E7 is trained on clips from hydrophone 2523 only (27,001 clips) and tested on the fold 1 test clips from hydrophone
2556, plus background (1,267 clips, 29 % background). On exactly the same test clips:

| | All 1,267 clips | Ship clips only (905) |
|---|---|---|
| E2, trained on both hydrophones | 41.1 % | **32.6 %** |
| E7, trained on hydrophone 2523 only | 40.2 % | **25.9 %** |

On ship clips the change of hydrophone costs about 7 points. Part of that may come from E7 training on fewer clips
(27,001 against 34,286). E7 recognises background better (76 % against 62 %), which hides the drop in the
all-clips column. Temperature scaling still works: ECE 0.320 → 0.068.

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

### 4.2 A second architecture (MobileNetV3-Large)

MobileNetV3 is a network designed for phones: it splits each convolution into a cheap per-channel filter and a
1×1 mixing step. It has about as many parameters as EfficientNet-B0 (4.22 against 4.02 million), but needs roughly
half the computation per clip. (The first idea, "a smaller network memorises less", doesn't hold: it is smaller in
computation, not in parameters.) What the run tests:

- **Model selection:** whether the result depends on the architecture. If both land in the same place, the limit is
  the data (too few ships per class), not the network.
- **Efficiency:** the same score at half the computation would make it the better choice for a small device, such as
  a battery-powered listening buoy.

The first attempt crashed at the start because of a bug in our model code. MobileNetV3 outputs 1,280 features, but
the output layer was sized from timm's `num_features` (960). The bug was fixed (`src/models.py`), with a test that
now builds every backbone.

| | EfficientNet-B0 (E2) | MobileNetV3-Large |
|---|---|---|
| Best validation macro-F1 | **0.425** (epoch 5) | 0.381 (epoch 7) |
| Validation loss | rises from epoch 2 | rises from epoch 1 (1.67 → 2.86) |
| Training speed | 260 clips/s, 238 s per epoch | **401 clips/s**, 145 s per epoch |

**Result: no improvement** (the test set was not used). MobileNetV3 trains about 1.5 times faster but scores lower,
and it memorises the training ships in the same way. Both architectures hit the same wall, which again points to
the data rather than the network.

### 4.3 Shorter runs for the review deadline (`configs/fast.yaml`)

Early stopping after 3 epochs without improvement instead of 6, and at most 12 epochs. E2's best epoch was 5, so
this mainly removes epochs that didn't help. All overnight runs use it.

### 4.4 Ideas from a related project

A classmate's project uses similar data: 10-s clips at 48 kHz from two hydrophone stations, with ships up to
18 km away. It predicts distance as 11 classes of 1 km, and vessel type for ships within 3 km. Its split is by
recording day, not by ship: 316 of its 589 test ships also appear in training (as in our E5), so its scores are
optimistic compared with ours. Its best distance results (validation): right 1-km bin 31.5 %, right bin or the next
one about 70 %, near/far (3 km) balanced 79 %.

| Idea | What we did | Result |
|---|---|---|
| Distance as classes | every distance evaluation also scores 250 m bins, "right bin or the next one" and near/far | 3.10: the same picture as the regression scores |
| MFCCs (20, mean and std), 10th / 50th / 90th percentile of every band and MFCC, how fast the MFCCs change | `baseline_classical.py --features rich` (761 numbers per clip instead of 257) | 5 folds: 29.5 ± 9.5 % against 28.4 ± 9.1 % (macro-F1 0.281 against 0.271); better on 4 of 5 folds (+2.2, +0.7, +0.6, −0.3, +2.1 points) |
| Rare vessel types grouped as "Other" | section 4.5 | 9 classes: 36.7 % (CNN) |

Not taken over: its 10 Hz–12 kHz filter (our spectrogram already starts at 10 Hz, and the filter didn't help there
either), frame-by-frame feature values and BioLingual embeddings (neither beat its simple features).

The richer features help the baseline by about 1 point on average (up to 2 points on some folds). This again says
that the limit is the number of different ships per class, not how the sound is summarised.

### 4.5 Few-ship classes merged into "other" (9 classes, decided 28 October)

Every class has about 2,870 training clips, but some come from very few ships: pilot vessel 2 ships in the whole
dataset, dredger 6, rescue 23. A class learned from one or a few ships can't be recognised on new ships (3.1, 3.8).
These three are now trained and scored as "other" (`--merge-into-other pilotvessel dredger rescue`), giving 9 classes.

Before training, an estimate from E2's test predictions (adding up the merged classes' probabilities) gave 34.2 %.
The trained model (`class_f1_efficientnet_b0_vessel_merged`, 10 epochs with `configs/fast.yaml`, best validation
macro-F1 0.430 at epoch 7) does a little better:

| Test set, fold 1 | 12 classes (E2) | **9 classes (E2-9)** |
|---|---|---|
| Accuracy, every clip | 32.4 % | **36.7 %** |
| Macro-F1 | 0.290 (11 classes in test) | 0.370 (9 classes) |
| Random guess | 8.3 % | 11.1 % |
| Ships < 700 m / 700–1,400 m / > 1,400 m | 43 / 33 / 22 % | **48** / 35 / 26 % |
| True class among the 3 most likely | 66.4 % | **70.5 %** |
| Most confident 20 % / 10 % of clips | 51.8 / 57.7 % | **57.8 / 65.1 %** |
| Per ship, clips averaged (217 ships with ≥ 3 clips) | 40.1 % | **41.9 %** |
| ECE before → after temperature scaling | 0.330 → 0.093 | 0.374 → 0.107 (T = 2.55) |

(Per ship here averages the temperature-scaled probabilities, for both models; section 3.6's 37.8 % used a slightly
different script.)

Recall per class: background 0.76, cargo 0.59, tanker 0.56, tug 0.46, pleasure craft 0.29, fishing 0.28, sailing 0.28,
passenger ship 0.24 and **other 0.11**. The enlarged "other" class is now the largest in the test set (873 of 4,477
clips) and the hardest: it mixes many kinds of ships, so it has no common sound to learn. The main confusions between
the large classes remain.

**The classical baseline on the same 9 classes:**

| 9 classes, test set | Fold 1 accuracy | Fold 1 macro-F1 | 5 folds |
|---|---|---|---|
| E1-9, 257 features | 34.6 % | 0.349 | 33.6 ± 6.6 % / 0.333 ± 0.050 |
| E1-9 rich, 761 features | 36.4 % | **0.371** | (fold 1 only) |
| E2-9 CNN | **36.7 %** | 0.370 | (fold 1 only) |

With the richer features, gradient-boosted trees on hand-made statistics are **as good as the CNN** (36.4 % against
36.7 %, macro-F1 0.371 against 0.370). On ships it has never heard, the CNN brings no advantage on this data.

Merging also makes the folds more alike: the baseline's fold 2 rises from 12.4 % to 23.0 % (the pilot vessel that
fills 63 % of its test set is now "other"), fold 4 from 31.4 % to 41.1 % (dredger), and the spread over folds falls
from ± 9.1 to ± 6.6 points.

### 4.6 Broader groups: estimates only, to discuss with the professor after the review

Same method as 4.5, from E2's fold 1 test predictions. Several groupings were tried on this test set, so these numbers
are optimistic. A grouping must be fixed before the final evaluation, and then trained and tested once.

| Grouping | Classes | Every clip | Ships < 700 m | Random guess |
|---|---|---|---|---|
| 12 types (now) | 12 | 32.4 % | 43 % | 8 % |
| pilot vessel, dredger, rescue → other | 9 | 34.2 % | 45 % | 11 % |
| pilot vessel, dredger, rescue and other left out | 8 | 43.1 % | 53 % | 13 % |
| **Size groups as in the ShipsEar dataset** ("other" left out) | 5 | **65.1 %** | **84 %** | 20 % |

The size groups, after the public ShipsEar dataset: A working and fishing vessels (fishing, tug, dredger), B small
craft (pleasure craft, sailing, pilot vessel, and rescue, placed by size), C passenger ships, D large ships (cargo,
tanker), E no ship (background). Weak spots remain: passenger ships 27 %, background 42 %. The placement of tugs and
pilot boats follows the original ShipsEar paper and should be checked there.

Both 5-class groupings were trained on 28 October: section 4.7.

### 4.7 Five classes, trained (28 October)

Two 5-class models with the same network and settings as E2-9 (fold 1, `configs/fast.yaml`); only the labels differ
(`--grouping`, groupings in `configs/groups.yaml`). Classes in no group are left out of training, validation and test.

- **E2-5S, size groups as in ShipsEar** (4.6): working and fishing (fishing, tug, dredger), small craft (pleasure
  craft, sailing, pilot vessel, rescue), passenger ships, large ships (cargo, tanker), no ship (background). "Other"
  is left out: it mixes many kinds of ships, so it fits no size group.
- **E2-5U, the 5 classes of the dataset's paper** (4.8): background, cargo, passenger ship, tanker, tug; the other 7
  classes are left out. It is the task the paper reports, so the two can be compared.

| Test set, fold 1 | E2-9 (9 classes) | **E2-5S size groups** | **E2-5U paper's classes** |
|---|---|---|---|
| Test clips | 4,477 | 4,052 | 2,250 |
| Accuracy, every clip | 36.7 % | **63.5 %** | **62.8 %** |
| Macro-F1 | 0.370 | 0.604 | 0.632 |
| Estimate before training, from E2's predictions | 34.2 % | 65.1 % (4.6) | 60.9 % |
| Always answering the largest class | 19.5 % (other) | 32.9 % (small craft) | 23.4 % (tanker) |
| Random guess | 11.1 % | 20 % | 20 % |
| Ships < 700 m / 700–1,400 m / > 1,400 m | 48 / 35 / 26 % | 70 / 68 / 53 % | 64 / 60 / 50 % |
| True class among the 2 most likely | 57.2 % | 86.9 % | 89.2 % |
| Most confident 20 % / 10 % of clips | 57.8 / 65.1 % | 91.2 / 96.5 % | 89.8 / 97.8 % |
| Per ship, clips averaged (ships with ≥ 3 clips) | 41.9 % (217) | 70.1 % (197) | 77.9 % (86) |
| ECE before → after temperature scaling | 0.374 → 0.107 | 0.110 → 0.031 (T = 1.50) | 0.184 → 0.026 (T = 2.25) |
| Training | 10 epochs (best 7), 42 min | 5 epochs (best 2), 19 min | 9 epochs (best 6), 16 min |

Recall per class: E2-5S large 0.83, background 0.82, small craft 0.64, working 0.48, passenger 0.35; E2-5U background
0.89, tug 0.67, tanker 0.66, cargo 0.51, passenger ship 0.44.

Main confusions (share of each true class): in E2-5S, passenger ships are taken for small craft (31 %) or working
vessels (24 %), working vessels for small craft (26 %), and small craft and background for each other (14 % each way:
a quiet small boat sounds like no ship). In E2-5U, cargo is taken for tanker 43 % of the time and tanker for cargo
27 %, the same main confusion as in the paper (4.8); passenger ships are taken for tugs 31 %.

What this shows:

- **Both pass the 60 % aimed for,** on ships never heard in training. Going from 9 to 5 classes removes the hardest
  choices: E2-5S puts cargo and tanker in one group, which removes the largest confusion, and E2-5U drops the classes
  with few ships.
- **E2-5S is slightly optimistic:** its grouping was picked after estimates on this test set (4.6). E2-5U's classes
  were fixed by the dataset's authors before we saw any result, so it is the cleaner number.
- **The estimates were close** (65.1 → 63.5 %, 60.9 → 62.8 %): a model trained on the 5 classes does about as well as
  the 12-class model asked the same question.
- **Passenger ships stay the weakest class** in both: 13 test ships in fold 1, 110 in the whole dataset.
- **Accuracy still falls with distance,** from 70 % to 53 % beyond 1.4 km in E2-5S.
- **The best epoch is early again** (2 and 6): the models soon start to memorise the training ships (3.2).
- **Calibration is much better than with 9 classes:** an ECE of 0.03 after temperature scaling.
- **E2-5S is fold 1 only.** The baseline's spread across folds was ± 6.6 to 9.5 points (3.8), so a few points'
  difference is not meaningful. E2-5U was run on all 5 folds (below).

**E2-5U on all 5 folds** (13:13–14:22, same settings; ± is the standard deviation over the folds, as for E1):

| Fold | Test clips (background share) | Accuracy | Balanced accuracy | Macro-F1 | Per ship (ships with ≥ 3 clips) | Epochs (best) |
|---|---|---|---|---|---|---|
| 1 | 2,250 (16 %) | 62.8 % | 63.5 % | 0.632 | 77.9 % (86) | 9 (6) |
| 2 | 953 (39 %) | 73.1 % | 67.8 % | 0.660 | 74.1 % (27) | 12 (12) |
| 3 | 1,661 (22 %) | 67.2 % | 66.4 % | 0.653 | 75.9 % (58) | 11 (8) |
| 4 | 1,535 (24 %) | 68.3 % | 67.1 % | 0.663 | 63.0 % (54) | 6 (3) |
| 5 | 1,887 (19 %) | 62.4 % | 62.7 % | 0.615 | 66.2 % (68) | 8 (5) |
| **Mean ± std** | | **66.8 ± 4.4 %** | **65.5 ± 2.3 %** | **0.644 ± 0.020** | 71.4 ± 6.5 % | |

- **Balanced accuracy** (the mean of the per-class recalls, every class counting equally) is the fairer number here:
  background, the easiest class (recall 0.87–0.97), makes up 16–39 % of our test sets, which lifts fold 2's
  accuracy to 73 %. The paper's test sets are balanced by duration.
- **Mean recall per class:** background 0.92, tug 0.75, cargo 0.67, tanker 0.51, passenger ship 0.44. Tanker and
  passenger ships are the weak classes on every fold.
- Fold 2 stopped at the 12-epoch limit of `fast.yaml` while still improving. ECE after temperature scaling:
  0.026–0.071.

### 4.8 The dataset's own paper (UniqueShip, read 28 October)

Our data is the "UniqueShip (12 Class – 5h Each)" subset of UniqueShip (Hashemi, Stout, Hoogs and Parham,
"UniqueShip: Mitigating Data Leakage in Acoustic Ship Classification Benchmark Datasets", OCEANS 2026,
arXiv 2609.13659; https://uniqueshipdata.org/). The paper's ship counts match ours exactly (110 passenger ships,
2,522 cargo, 297 tug, 233 tanker; 4,218 ships), and the fold CSVs in our zip are its leak-free folds: ships grouped
by MMSI, background grouped by day. The paper's clips are resampled to 20 kHz; our zip holds them at 32 kHz.

The paper reports results only for 5 classes (background, cargo, passenger ship, tanker, tug), on its
"5 Class – Balanced" set with 42 h 40 min of audio per class, 8.5 times our 5 h:

| Without leakage, 5 classes | Audio per class | Accuracy | Macro-F1 |
|---|---|---|---|
| Paper: MobileNetV3-Small, Mel spectrogram (CNN) | 42.7 h | 61.7 ± 1.2 % | 0.607 ± 0.018 |
| Paper: ViT-B/16, Mel (transformer) | 42.7 h | 65.0 ± 2.7 % | 0.640 ± 0.035 |
| Paper: SwinV2-Tiny, Mel (transformer) | 42.7 h | 66.5 ± 1.2 % | 0.656 ± 0.017 |
| Ours: E2-5U, EfficientNet-B0 (CNN), fold 1 | 5 h | 62.8 % | 0.632 |
| **Ours: E2-5U, mean of 5 folds** | **5 h** | **66.8 ± 4.4 %** (balanced 65.5 ± 2.3 %) | **0.644 ± 0.020** |

The paper's numbers are means over the 5 folds of its "5 Class – Balanced" set. Ours are over the 5 folds of our
12-class subset, keeping only these 5 classes, so the test ships are not the same, and our smaller test sets give a
larger spread. Within these spreads, our CNN is level with the paper's two transformers and above its MobileNet.
Possible reasons, not tested: our spectrogram keeps more low-frequency detail (32 kHz audio, a 4,096-sample window,
128 Mel bands from 10 Hz) than theirs (20 kHz audio, a 1,024-sample window, resized images); EfficientNet-B0 is a
larger CNN than MobileNetV3-Small; our learning rate is much higher (3e-4 against 1e-6).

What it means for us:

- **More hours of the same ships would help little.** The bigger releases (65 GB for the 5 classes, 1.4 TB for
  everything) contain the same ships as ours, only more hours of each. In the paper's tests, doubling the audio of
  the same ships adds 0.8–1.3 points, and doubling the number of different ships adds 2.4–2.6. 8.5 times the audio
  is about 3 doublings, so about +2.5 to +4 points: not worth a 65 GB download on this laptop. By the same rule, the
  paper's best model would be expected at about 62–64 % with our 5 h per class; E2-5U's 5-fold mean (balanced
  accuracy 65.5 %) is at or above that.
- **The 12-class task has no published reference:** the paper's 66.5 % is for 5 classes.
- **The same findings as ours:** leakage inflates accuracy (66.5 → 82.6 % for their Swin; 32 → 86 % for our E2 and
  E5), the models key on the individual ship (in their metadata analysis, knowing the MMSI best predicts which clips
  are classified correctly; our 3.7), and cargo against tanker is the main confusion.
- **One difference:** they find that distance on its own hardly predicts which clips are classified correctly within
  the 2 km radius, while our accuracy falls clearly beyond about 1.4 km (3.4, 4.7). Their test (predicting
  correctness from metadata) is different, so the two don't directly conflict.
- **Their recipe differs from ours:** ImageNet-pretrained backbones with a two-layer head, spectrograms with a 90 dB
  range resized to 224 × 224 or 256 × 256, and AdamW with a learning rate of 1e-6 (as printed) for 30 epochs. Mel
  spectrograms always beat plain STFT, and the transformers beat MobileNet by 3–5 points.

### 4.9 Six classes, interpretable features, two classical models (1 November)

Decided on 1 November: the earlier work had grown into many models and variants, and the ship counts per class were
very unequal. The new, smaller question (`src/six_classes.py`, `notebooks/13_six_classes_and_features.ipynb`):

- **Classes:** background, cargo, fishing, pleasure craft, tanker, tug. Passenger ship is dropped (110 ships); every
  class kept has at least 233 ships (background: recording days). Each class has the same 5 h (about 3,600 clips).
- **Balancing:** hours are already equal; ship counts are not (cargo 2,522 ships with 1–2 clips each, tanker 233 with
  about 15). No clip is deleted: training weights give every class the same total and every ship the same share
  within its class. Inside each class the clips per ship hardly vary, so this is close to plain class weights.
- **Folds:** 5 new folds (`StratifiedGroupKFold`, grouped by MMSI, background by day): every test fold holds a fifth
  of every class's ships (tanker 46–47 per fold, against 10–34 in the official folds), and every ship is tested once.
  Per run: one fold test, the next validation (to choose C or the number of trees), three training; then refit on
  training + validation.
- **Not downloaded:** the 64 GB `main5_balanced_audio` set: the same ships as ours, more hours of each, and no
  fishing or pleasure craft.
- **Models:** logistic regression (standardised, C from 0.01 / 0.1 / 1) and gradient boosting (early stopping on the
  validation ships). No CNN, no leaky-split or hydrophone tests.
- **Simple features ("normal model"):** mean and spread of each mel band + overall mean (257, as E1), no MFCC.
- **Engineered features (144), from the raw audio:** MFCC 1–19 (mean, spread, change; 57), 1/3-octave band levels
  20 Hz–12.5 kHz (29), level variation per octave (9), spectral shape per frame: centroid, bandwidth, roll-off,
  flatness, flux, energy, zero-crossing rate, each as mean / std / 10th / 90th percentile (28), loudness and
  impulsiveness: broadband level, spectral slope, low minus high level, kurtosis, crest factor (5), tonal lines below
  1 kHz (LOFAR, 5) and propeller modulation of the 2–10 kHz noise (DEMON, 11). They replace the 761 "rich" numbers of
  section 4.4. Extraction: 10 min for 21,604 clips (4 processes); the 5-fold run: 19 min, CPU.

| 6 classes, 5 new folds (± over folds) | Accuracy | Balanced accuracy | Macro-F1 | Per ship (balanced) |
|---|---|---|---|---|
| simple + logistic regression | 56.9 ± 1.3 % | 56.9 ± 1.3 % | 0.564 ± 0.013 | 64.9 ± 2.5 % |
| simple + gradient boosting | 58.4 ± 1.2 % | 58.4 ± 1.1 % | 0.579 ± 0.012 | 66.6 ± 2.8 % |
| engineered + logistic regression | 60.8 ± 1.3 % | 60.8 ± 1.3 % | 0.604 ± 0.014 | 67.3 ± 1.8 % |
| **engineered + gradient boosting** | **61.9 ± 1.1 %** | **61.9 ± 1.1 %** | **0.614 ± 0.012** | **70.5 ± 2.2 %** |

Chance is 16.7 %. Accuracy equals balanced accuracy because the new test folds hold about the same number of clips
per class. Recall per class (engineered + gradient boosting): background 85 %, cargo 65 %, pleasure craft 64 %, tug
62 %, tanker 52 %, fishing 43 %.

What we found:

- **Engineered features beat the simple ones by 3.5–3.9 points with 144 numbers instead of 257**, on every fold for
  gradient boosting. They help most for tug (+7 points), cargo (+5), pleasure craft (+5) and fishing (+3.5).
- **The higher MFCCs help** (MFCC group alone, validation): 4 → 8 → 12 → 16 → 19 coefficients give 44.5, 46.8, 48.2,
  48.7 and 49.3 % (mean of both models), so all 19 are kept. The expectation that they add nothing did not hold.
- **The folds agree within ± 1.1–1.3 points**, against ± 4.4 (E2-5U) and ± 9.1 (E1) on the official folds.
- **Main confusion: cargo against tanker** (30 % of cargo called tanker, 43 % of tanker called cargo): their median
  spectra lie on top of each other. Fishing is confused with tug (25 %) and pleasure craft (19 %).
- **Feature groups** (alone, gradient boosting / drop when shuffled in the full model): 1/3-octave levels 54.5 % / 24
  points, spectral shape 53.8 % / 5, MFCC 52.4 % / 7, loudness and impulsiveness (5 numbers) 49.3 % / 2, DEMON
  40.8 % / 4, tonal lines 39.1 % / 1.5, level variation 37.7 % / 3.
- **No hydrophone shortcut** (every ship class is 79–82 % hydrophone 2523). Cargo and tankers pass closer (median
  about 900 m against 1,350–1,500 m), worth about 4 dB of their 20–30 dB higher level at 50 Hz. Balanced accuracy on
  ship clips: 61 % within 1,400 m, 52 % beyond.
- For comparison only (different folds): E2's 12-class predictions limited to these 6 classes give about 57 %
  balanced accuracy on fold 1.

Limits: one seed; levels uncalibrated; the number of MFCCs was chosen on the validation folds of all five runs (each is
another run's test fold), a small look at test data for that one choice; new folds, so not comparable number for number
with the paper or with sections 2–4.8.

---

## 5. Engineering notes from the training runs

- **Heat:** the GPU reached 86–87 °C every epoch. The temperature guard paused training until it cooled below
  75 °C, costing about 45–105 s per 4-minute epoch.
- **Memory:** the first E2 evaluation was stopped when the laptop ran low on RAM, while other programs (browser,
  IDE, chat apps) were also open. Training was already saved, so only the evaluation was rerun. Long runs now start
  in a separate window.
- **One job at a time:** at 02:10 on 28 October, unknown-ship group C crashed in epoch 2 with a GPU error
  (`cuDNN error: CUDNN_STATUS_EXECUTION_FAILED`; the temperature read failed just before). At the same time, classical
  baselines and the test suite were running on the CPU, and free RAM had dropped to 2 GB: those jobs failed with
  out-of-memory errors themselves. CPU jobs now wait until GPU training has finished, and the classical features are
  computed 250 clips at a time. Group C resumes from its saved epoch.
- **Where the memory goes:** with `--workers 4` a training run holds about 6 GB in the main process plus about 2 GB in
  each data-loading helper (4 for training and up to 4 for validation). On Windows every helper loads its own copy of
  PyTorch's GPU libraries. Together that is more than the laptop's 16 GB, so Windows uses the page file. `--workers 2`
  should roughly halve the helpers' share; the GPU, not data loading, limits the speed here.
- **Resuming:** every epoch is saved (`last.pt`), so a crash costs at most one epoch.

---

## 6. Limitations

- **One fold for the CNN, one random seed.** Differences of a few hundredths in macro-F1 are not meaningful. The
  baseline has 5 folds (3.8). E2 on all 5 folds (about 3.5 h) was left out because of the review deadline. The
  5-class E2-5U has all 5 folds (4.7).
- **Few ships in rare classes:** pilot vessel 2, dredger 6 and rescue 23 in the whole dataset. Their scores rest on
  one or a handful of ships, and they dominate some folds' test sets (3.8).
- **Validation and test differ in class mix** (3.3).
- Only two hydrophones at one location, one ship at a time, ships up to about 2 km away.
- **The 5 size groups were chosen after estimates on the same test set** (4.6), so E2-5S is slightly optimistic;
  E2-5U's classes were fixed by the dataset's authors.

---

## 7. Run overnight (27–28 October)

| Time | Run |
|---|---|
| 23:47–00:34 | distance (E9) and speed (E10), each with its evaluation |
| 00:34–01:12 | unknown ships group A + evaluation (MC dropout); noise test on E2 (E8) |
| 01:12–01:24 | baseline, folds 2–5 |
| 01:24–02:10 | unknown ships group B + evaluation; group C crashed in epoch 2 (section 5) |
| 02:10–03:02 | hydrophone shift (E7) + evaluation; MobileNetV3 |
| 03:03–03:46 | **9-class CNN (E2-9)** + evaluation |
| 03:46–04:22 | unknown ships group C, resumed from epoch 1, + evaluation |
| 04:22–04:50 | baselines: 9 classes (folds 1–5, rich features on fold 1), rich features on folds 3–5 |
| 04:50–04:53 | lessons 03–10 executed in place, so they show these results; lesson 10 collected the figures in `runs/report/` |

Left out for time: the 3-model deep ensemble, and the CNN on folds 2–5.

Daytime, 28 October: E2-5S 12:04–12:24, E2-5U 12:24–12:41 and E2-5U folds 2–5 13:13–14:22, each with its
evaluation (section 4.7).

---

## 8. Ideas not yet tried

These follow from 3.7. None has been tested.

- **Ship-level prediction:** report a decision per ship passage (averaging clips, 3.6), as in real use.
- **Answer only when confident** (3.12), with the threshold chosen on validation, reporting accuracy and the share
  of clips answered.
- **Augmentation that changes ship identity but keeps the class:** mixup between ships of the same class, pitch or
  speed shifts, mixing in recorded background noise.
- **Class-balanced sampling by ship,** so that ships with many clips don't dominate training.
- **Broader groups** (4.6): two groupings trained on 28 October (4.7); which one to use is to agree with the
  professor.
- **Gentler fine-tuning, as in the dataset's paper** (4.8): a learning rate of 1e-6 (as printed) for 30 epochs,
  against our 3e-4; our runs peak at epochs 2–7 and then memorise the training ships.
- **Transformer backbones** (4.8): the paper's best models are SwinV2-Tiny and ViT-B/16 on Mel spectrograms, 3–5
  points above its MobileNet; both are available in timm.
- **Distance as a bin classifier,** which would also give a confidence for the distance.
- **5-fold evaluation** of the final model, to report a mean ± standard deviation (done for E2-5U, 4.7).

---

## 9. Reproduce

```
python src/baseline_classical.py --fold 1                                      # E1 (--fold 2..5 for the other folds)
python src/baseline_classical.py --fold 1 --features rich                      # E1 rich
python src/train.py --task class                                               # E2
python src/evaluate.py runs/class_f1_efficientnet_b0_vessel --mc 20
python src/evaluate.py runs/class_f1_efficientnet_b0_vessel --noise 20 10 0    # E8 (needs the audio zip)
python src/train.py --task class --config configs/strong_regularization.yaml --name class_f1_efficientnet_b0_vessel_reg
python src/train.py --task class --split-mode leaky                            # E5
python src/evaluate.py runs/class_f1_efficientnet_b0_leaky
python src/train.py --task class --holdout A --config configs/fast.yaml        # E3 (also B, C)
python src/evaluate.py runs/class_f1_efficientnet_b0_vessel_hA --mc 20
python src/train.py --task distance --init-from runs/class_f1_efficientnet_b0_vessel/best.pt --config configs/fast.yaml   # E9
python src/train.py --task speed --init-from runs/class_f1_efficientnet_b0_vessel/best.pt --config configs/fast.yaml      # E10
python src/train.py --task class --merge-into-other pilotvessel dredger rescue --config configs/fast.yaml                 # 9 classes
python src/baseline_classical.py --fold 1 --merge-into-other pilotvessel dredger rescue
python src/train.py --task class --grouping shipsear5 --config configs/fast.yaml                                         # E2-5S
python src/train.py --task class --grouping uniqueship5 --config configs/fast.yaml                                       # E2-5U (--fold 2..5 too)
python src/evaluate.py runs/class_f1_efficientnet_b0_vessel_shipsear5                                                    # (also _uniqueship5)
python src/six_classes.py features                                                                                       # 6 classes (4.9): features from the raw audio
python src/six_classes.py run                                                                                            # 6 classes: 5 folds, both feature sets, both models
```
