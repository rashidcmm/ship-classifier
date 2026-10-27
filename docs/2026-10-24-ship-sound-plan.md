# Reliability-Aware Underwater Vessel Recognition — Project Plan

**Status:** v3 (approved: "continue your work"). Merges your presentation ("Reliability-aware underwater acoustic vessel recognition") with the distance and speed objectives. **Training happens on a different laptop**, specs unknown (see section 11) · **Date:** 2026-10-25

## 1. Goal

Given a **new 5-second underwater recording**, the system should:

1. **Recognise the vessel class** (11 ship types + background), **on ships it has never heard before**.
2. **Say "unknown"** instead of forcing a wrong answer when the sound doesn't match any known class, and give a **confidence you can trust**.
3. Estimate **how far** the ship is from the hydrophone.
4. Estimate **how fast** the ship is going.

…and then **prove it** with fair tests (sections 5–6), as promised on the presentation's last slide.

**Your learning goal:** you want to understand every step: how data is set up, how a CNN is trained, how it is tuned, and how algorithms are chosen. So everything is delivered as **numbered lessons** (section 8), not only as scripts.

**Who does what:** I set up the environment, data, code and lessons, and run only tiny tests. **You run the real training.**

**Not in scope:** direction or position of the ship, several ships at once, ships beyond 2 km, chasing leaderboard scores.

**Failure even if it "works":**
- High accuracy that comes from the same ship being in training and test (leakage). Prevented by the dataset's vessel-disjoint folds.
- A model that calls new ships of a *known* type "unknown" all the time. That would be useless to a sonar operator.
- Damaging the laptop (section 10).

---

## 2. The data (measured)

| Item | Value |
|---|---|
| Clips | 43,208 × 5 s (39,608 ship + 3,600 background) |
| Audio | mono, **32 kHz** (the presentation's slide 4 says 20 kHz; the files are 32 kHz, so fix this before the final report) |
| Ships | 4,218 vessels; 12 labels (11 ship types + background); 2 hydrophones |
| Distance `mean_distance` | 7.7 – 2,000 m, median 1,295 m (99.7 % of ship clips) |
| Speed `mean_sog` | 0.1 – 38.6 knots, median 9.6 (100 % of ship clips) |
| Folds | 5 vessel-disjoint train/validation/test splits; fold 1 = 34,286 / 4,445 / 4,477 |

**The bar to beat (fold 1 test):** class ≈ 8 % (random guess out of 12) · distance error 403 m (always guessing 1,295 m) · speed error 3.6 knots (always guessing 9.6).

**Rare classes (your decision: keep all 12).** `pilotvessel` has only 2 ships and `dredger` only 6. Fold 1's test set has 36 pilot-vessel clips and **no** dredger clips. Their per-class scores will be reported, with a warning that they aren't reliable.

**Provided spectrogram PNGs are not used.** Per-image scaling probably removes absolute loudness, which distance depends on. We compute our own log-mel features from the raw audio.

---

## 3. How "unknown" works (and how we test it)

There are two different kinds of "new":

| Situation | Example | Wanted answer |
|---|---|---|
| **New ship, known type** | a tanker the model never heard, but it was trained on other tankers | "tanker". **Not** unknown |
| **New type of ship** | the model never saw *any* fishing boat in training | "unknown" |

The vessel-disjoint folds already make every test ship a new ship (row 1).

**To test row 2, the unknown classes must be *missing from training* and *present in test*.** This is the reverse of "in training but not in test". We do this by **holding out whole classes**:

- Train the model on, say, 9 known classes. The 3 held-out ship classes are removed from training and validation.
- At test time, the held-out classes' clips should be flagged "unknown". All other test clips should get their correct class.
- Repeat with **3 different held-out groups**, so the result doesn't depend on one lucky choice. My proposed groups are below; you can edit them in `configs/open_set.yaml`.
  - group A: fishing, sailing, tanker
  - group B: cargo, pleasurecraft, rescue
  - group C: tug, passengership, other
  - background, pilotvessel and dredger stay "known" in all groups. The last two are too small to make a useful unknown test set.
- "Not a ship" sounds other than ocean background (whales, rain, engines on shore) are **not in this dataset**. We approximate them with the added-noise test in section 6, and say so in the report.

### Scoring "is it unknown?" (from your slide 4)

| Method | Idea | Extra training cost |
|---|---|---|
| Max softmax probability (MSP) | low top probability → unknown (the baseline to beat) | none |
| Energy score | uses the raw network outputs before softmax; often separates unknowns better | none |
| Monte Carlo dropout | run the model 20× with dropout on; disagreement → unsure | none (20× slower prediction) |
| Deep ensemble | train 3–5 models with different random starts; disagreement → unsure | 3–5× training |

### Making confidence trustworthy

- **Temperature scaling:** one number, fitted on the validation set, that turns "90 % confident" into "right about 90 % of the time".
- Measured with **ECE** (expected calibration error) and **reliability diagrams**.
- Unknown detection is measured with **AUROC** and **FPR at 95 % TPR**. Open-set accuracy uses a threshold chosen on validation, never on test.

---

## 4. How the system works

```
raw WAV clips (read straight from the downloaded zip — never unzipped)
   │ A. features, once (~10–20 min): log-mel 128 bands × ~500 steps, float16, ~5.5 GB, keeps true loudness
   ▼
   ├─ B. classical baseline (CPU, minutes): band statistics + gradient-boosted trees
   │
   └─ C. CNN (GPU): pretrained EfficientNet-B0 reading the spectrogram as an image
         C1 class (12-way, closed set)                → accuracy, per-class, confusion matrix
         C2 unknown + confidence                      → MSP / energy / MC dropout / ensemble, temperature scaling
         C3 distance (head added to the C1 model)     → error in m, near/mid/far
         C4 speed    (head added to the C1 model)     → error in knots, slow/medium/fast
   ▼
predict.py recording.wav → class or "unknown", confidence, distance, speed
```

- **Features:** keep absolute loudness (one global normalisation, never per clip). The hydrophone ID is *not* given to the model, because new recordings won't have it.
- **Distance / speed:** trained on ship clips only, with Huber loss (robust to bad GPS points), each as its own stage starting from the C1 model.
- **Limit:** distance is learned from loudness on *these* hydrophones. On another microphone or gain setting, class prediction can still work, but distance needs calibration.

---

## 5. Experiments (what the report will show)

| # | Experiment | Question it answers | Training runs (fold 1) |
|---|---|---|---|
| E1 | Classical baseline vs CNN | Is deep learning worth it here? | CPU only + C1 |
| E2 | Closed-set 12-class CNN | How well are known types recognised on unseen ships? | 1 |
| E3 | Unknown detection, groups A/B/C | Do energy / MC dropout / ensembles beat softmax? | 3 (+ ensembles) |
| E4 | Calibration before/after temperature scaling | Is the confidence trustworthy? | 0 (reuses E2) |
| E5 | **Leaky split** (random clips, same ship in train and test) vs vessel-disjoint | How much do leaky benchmarks inflate accuracy? | 1 |
| E6 | Accuracy vs distance band | Does recognition get worse for far ships? | 0 (reuses E2) |
| E7 | Train on hydrophone 2523, test on 2556 | Does it survive a different deployment? | 1 |
| E8 | Added noise at 20 / 10 / 0 dB SNR | Does it fail gracefully, and does it say "unknown"? | 0 (reuses E2) |
| E9 | Distance regression | How far is the ship? | 1 |
| E10 | Speed regression | How fast is it going? | 1 |

---

## 6. Order of work (matches your slide 5, plus distance and speed)

| Phase | Content | Lessons |
|---|---|---|
| 0. Setup (me) | environment, features, smoke test | 00 |
| 1. Baseline CNN | E1, E2, E4 | 01 – 06 |
| 2. Unknown-ship detection | E3 (+ ensembles) | 07 – 08 |
| 3. Distance and speed | E9, E10 | 09 |
| 4. Tests and report | E5 – E8, figures and tables | 10 |

---

## 7. Time estimates (RTX 3050 4 GB laptop)

Estimated about 1–2 min per epoch and 20–30 epochs per training run, so **~1–1.5 h per CNN training run**. The smoke test measures the real speed, and the estimates get updated then.

| Phase | GPU time |
|---|---|
| 1. Baseline CNN (E2) | ~1–1.5 h |
| 2. Unknown detection: 3 groups × 1 run | ~3–4.5 h |
| 2b. Deep ensemble: +2–4 extra models of E2 | ~2–6 h (**optional; good candidate for Kaggle**) |
| 3. Distance + speed | ~1.5–2 h |
| 4. Leaky split + hydrophone shift (2 runs) | ~2–3 h |
| **Total, fold 1** | **≈ 10–15 h**, spread over several sessions |
| All 5 folds (optional, for a final paper-style report) | ≈ 5× |

---

## 8. Learning path (lessons)

Each lesson is a **Jupyter notebook** that explains the idea in plain language, shows it on real data, and has small exercises. Long training runs use the `src/` scripts (a notebook dies if the window closes). The lessons show you how to run them and how to read the results.

| # | Lesson | You learn |
|---|---|---|
| 00 | Setup check | GPU, environment, where everything lives |
| 01 | Explore the data | CSV columns, classes, ships, folds, listen to clips, why vessel-disjoint splits matter |
| 02 | From sound to spectrogram | waveform, FFT, STFT, mel scale, decibels; how each parameter changes the picture |
| 03 | Classical baseline | features by hand, gradient boosting, train/val/test, accuracy vs macro-F1 — **algorithm selection** |
| 04 | CNN from scratch, tiny | convolution, pooling, layers; train a tiny CNN on a small subset in minutes; overfitting live |
| 05 | Training the real model | transfer learning, loss, optimizer, learning rate and schedule, batch size, augmentation (SpecAugment, mixup), mixed precision, checkpoints, early stopping |
| 06 | Tuning and model selection | what to tune and in what order, learning-rate finder, small sweeps on the **validation** set (never test), comparing EfficientNet-B0 / MobileNetV3 / ResNet-18, learning curves |
| 07 | Unknown-ship detection | open-set idea, MSP vs energy, held-out classes, AUROC / FPR@95 |
| 08 | Trustworthy confidence | calibration, ECE, reliability diagrams, temperature scaling, MC dropout, ensembles |
| 09 | Distance and speed | regression vs classification, Huber loss, error analysis, why distance depends on loudness |
| 10 | Stress tests and report | leaky split, distance bands, hydrophone shift, noise; making the final figures |

---

## 9. What gets set up (deliverables)

```
D:\shipsound\
  .venv\                  Python 3.12 environment: PyTorch (CUDA), timm, scikit-learn, pandas, matplotlib, soundfile, JupyterLab
  configs\                default settings + open_set.yaml (held-out groups)
  src\
    shipzip.py            read clips straight from the split zip
    features.py           log-mel spectrogram (written by hand, so lesson 02 can explain it)
    prepare_features.py   step A
    dataset.py            loads features + targets for any fold / task / held-out group
    models.py             EfficientNet-B0 / MobileNetV3 / ResNet-18 backbones + heads
    train.py              --task class|distance|speed  --fold N  --holdout A|B|C  --split vessel|leaky
                          --hydrophone-shift  --seed  --resume  --max-hours  (temperature guard built in)
    evaluate.py           all metrics + figures (closed set, open set, calibration, regression, noise)
    baseline_classical.py step B
    predict.py            new WAV → class / unknown, confidence, distance, speed
  notebooks\              lessons 00–10
  data\                   features + targets (created by step A)
  runs\                   one folder per training run: checkpoints, logs, metrics, figures
  README.md               every command, in order
```

**I run:** environment install, feature preparation, and a smoke test (1 epoch on ~200 clips, ~1–2 min, to measure real speed and prove the code works).
**You run:** everything in sections 5–7.

---

## 10. Laptop safety and working gradually

- **Training cannot "fuse" RAM or burn the motherboard.** The CPU and GPU slow themselves down when hot, and in an extreme case the laptop switches off. The load is like long gaming sessions.
- Expect GPU 70–85 °C and loud fans. `train.py` **pauses automatically above 85 °C** and resumes below 75 °C.
- RAM use ~3–6 GB of 16. GPU memory 4 GB: if it's exceeded you get an error, not damage, so lower `--batch-size`.
- **Battery is the only part that really wears:** keep the laptop plugged in and enable MyASUS → Battery Health Charging (80 %).
- **Checkpoint after every epoch:** stop anytime with Ctrl+C and continue with `--resume`. `--max-hours 1.5` makes training run in sessions. A power cut loses at most one epoch.
- **Kaggle** (free, ~30 GPU h/week, keeps running with the browser closed) is the backup, mainly for the ensembles and the 5-fold runs. Only the ~5.5 GB feature file needs uploading. The README will include a short guide.

---

## 11. Training on another laptop (portability)

The real training runs on **a friend's laptop**, not this one. Its operating system, GPU and memory are unknown, so:

- **Nothing machine-specific in the code.** All paths are relative to the project folder. The device is detected automatically (NVIDIA CUDA, Apple MPS, or CPU), and the batch size is chosen from the GPU's memory unless set by hand.
- **What gets copied over (~5.6 GB):** `src\`, `configs\`, `notebooks\`, `docs\`, `README.md`, `requirements.txt`, and `data\` (the finished feature file, ~5.5 GB). Copy by USB drive or Google Drive. The 37 GB raw download is **not** needed there, except for the noise test (E8) and `predict.py` on new audio from the dataset. Those need only the 16.6 GB audio zip, and are optional on that machine.
- **Setup there:** one setup script per OS creates the Python environment and installs the right PyTorch build for that GPU. Then `python src/check_setup.py` prints the GPU, its memory, and whether the data is found and intact (it compares checksums written now).
- **Speed on that machine:** `python src/benchmark.py` trains for ~1 minute and prints the time per epoch and the estimated hours for each experiment in section 5. This replaces the RTX 3050 estimates in section 7.
- **Same safety features:** keep-awake while running, a checkpoint every epoch, `--resume`, `--max-hours`, and the GPU temperature guard (NVIDIA only; skipped elsewhere).
- **This laptop** stays the place for preparing data, writing code and lessons, and quick checks. Its crashes happen in sleep, and the diagnosis is handled in a separate session.

## 12. Calls I made (override any)

- PyTorch + pretrained EfficientNet-B0 as the main model. MobileNetV3 (one of the dataset authors' baselines) and ResNet-18 are available for comparison in lesson 06.
- Log-mel: 128 bands, 10 Hz – 16 kHz, ~10 ms steps; our own features instead of the PNGs.
- Fold 1 for everything first; 5 folds optional.
- Held-out groups A/B/C as listed. Pilot vessel and dredger are never used as unknowns.
- Thresholds for "unknown" chosen on validation data only.
- Huber loss for distance and speed; near/mid/far (0–700 / 700–1,400 / 1,400–2,000 m) and slow/medium/fast (< 6 / 6–12 / > 12 knots) bands.
- Hydrophone ID not used as model input.
- Lessons as Jupyter notebooks; long runs as scripts.
