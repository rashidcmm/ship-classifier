# Ship Sound Recognition — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Portable PyTorch code, plus lessons, that train and evaluate a CNN on UniqueShip log-mel features for vessel class, "unknown" detection with trustworthy confidence, distance and speed, ready to copy to another laptop and run there.

**Architecture:** Features already exist (`data/mel.npy` float16 (43208,128,501) + `data/clips.parquet`). Small focused modules in `src/`: data selection, models, metrics and reliability are pure and unit-tested on a synthetic mini-dataset. Thin CLIs (`train.py`, `evaluate.py`, …) compose them. Lessons are percent-format `.py` files built into `.ipynb` with nbformat.

**Tech Stack:** Python ≥3.10 (3.12 here), PyTorch 2.x (CUDA/MPS/CPU), timm, numpy, pandas+pyarrow, scikit-learn, scipy, matplotlib, soundfile, pyyaml, tqdm, pytest, JupyterLab/nbformat/nbclient.

**Spec:** `docs/2026-10-24-ship-sound-plan.md` (v3). Read sections 3, 4, 5, 9, 11 before starting.

## Global Constraints

- No absolute paths in code. Everything resolves from `src/paths.py`. The env vars `SHIPSOUND_DATA` and `SHIPSOUND_AUDIO_ZIP` override the data dir and the audio zip.
- Must run on Windows, Linux and macOS. Device order: CUDA → MPS → CPU. AMP only on CUDA.
- Class order is fixed everywhere: `["background","cargo","dredger","fishing","other","passengership","pilotvessel","pleasurecraft","rescue","sailing","tanker","tug"]`.
- Open-set groups (`configs/open_set.yaml`): A = fishing, sailing, tanker · B = cargo, pleasurecraft, rescue · C = tug, passengership, other. Background, pilotvessel and dredger are never held out.
- The hydrophone ID is never a model input.
- Thresholds, the temperature and early stopping use the **validation** split only. The test split is used only for final metrics.
- Distance target = `mean_distance / 1000` (km). Speed target = `mean_sog / 10`. Reported in m and knots. Huber loss with δ = 0.2 for both.
- Bands: distance near/mid/far = [0,700) / [700,1400) / [1400,2000] m. Speed slow/medium/fast = [0,6) / [6,12) / [12,∞) knots.
- Every long-running CLI (train, evaluate, baseline, benchmark) runs inside `keepawake.keep_awake()`.
- Checkpoints are written atomically (write `*.tmp`, then `os.replace`).
- `D:\shipsound` is **not a git repository**, so commit steps are omitted. Verification steps replace them.
- Test command (from the project root): `.venv\Scripts\python -m pytest <target> -v` (Windows) / `.venv/bin/python -m pytest …`.

## Review Focus

1. **A machine without an NVIDIA GPU** (CPU or Apple MPS, no `nvidia-smi`): the temperature guard no-ops, AMP is off, and the batch size is smaller. Test in Task 3: `test_device_without_nvidia_smi`.
2. **A run killed during a checkpoint write, then `--resume`:** `last.pt` is never half-written, and resume continues at the next epoch with the same best metric. Test in Task 5: `test_resume_after_kill`.
3. **No internet on the friend's laptop for pretrained weights:** a clear error naming `--no-pretrained`, not a raw HTTP traceback. Test in Task 3: `test_pretrained_download_failure_message`.
4. **Real-world WAVs given to `predict.py`:** 48 kHz stereo, 12 s long, or only 3 s long. Must resample, downmix, window and pad. Test in Task 8: `test_predict_odd_wavs`.
5. **A missing or partially copied `data/`** (e.g. a truncated `mel.npy`): `check_setup.py` and `train.py` fail with a plain message saying what to copy. Test in Task 9: `test_check_setup_truncated_mel`, and Task 2: `test_open_features_truncated`.

---

## File map

| File | Responsibility |
|---|---|
| `src/paths.py` | ROOT, DATA_DIR, RUNS_DIR, CONFIG_DIR, AUDIO_ZIP |
| `src/labels.py` | CLASSES, open-set groups, known classes |
| `src/data.py` | open features, row selection, normalisation, `MelDataset`, SpecAugment |
| `src/device.py` | device choice, auto batch size, GPU temperature, `TemperatureGuard` |
| `src/models.py` | `TinyCNN`, `ShipNet`, checkpoint save/load |
| `src/metrics.py` | classification, calibration, open-set, regression metrics (pure numpy) |
| `src/reliability.py` | MSP, energy, entropy, MI, temperature fit, MC dropout, ensembles |
| `src/train.py` | training CLI |
| `src/evaluate.py` | evaluation CLI + figures |
| `src/baseline_classical.py` | band-statistics + gradient boosting baseline |
| `src/predict.py` | WAV → class / unknown / distance / speed |
| `src/check_setup.py`, `src/benchmark.py` | target-machine checks and speed estimate |
| `configs/default.yaml`, `configs/open_set.yaml` | hyperparameters, held-out groups |
| `setup/setup_windows.ps1`, `setup/setup_unix.sh`, `requirements.txt`, `pytest.ini`, `README.md` | portability |
| `tests/conftest.py`, `tests/test_*.py` | synthetic mini-dataset + unit tests |
| `notebooks/lessons/NN_*.py`, `notebooks/build_notebooks.py`, `notebooks/NN_*.ipynb` | lessons |

Existing and kept: `src/shipzip.py`, `src/features.py`, `src/prepare_features.py`, `src/keepawake.py`.

---

### Task 1: Scaffolding, labels, configs, test fixture

**Files:**
- Create: `src/paths.py`, `src/labels.py`, `configs/default.yaml`, `configs/open_set.yaml`, `requirements.txt`, `pytest.ini`, `tests/conftest.py`, `tests/test_labels.py`
- Modify: `src/prepare_features.py`. `DEFAULT_ZIP` → `paths.AUDIO_ZIP`, `--out` default → `paths.DATA_DIR`

**Interfaces:**
- Produces:
  - `paths.ROOT: Path` (parent of `src/`), `paths.DATA_DIR`, `paths.RUNS_DIR = ROOT/"runs"`, `paths.CONFIG_DIR = ROOT/"configs"`, `paths.AUDIO_ZIP` (default `ROOT/"gdrive_download/audio/allclass_5hours.zip"`)
  - `labels.CLASSES: list[str]` (the order from Global Constraints)
  - `labels.load_groups(path: Path | None = None) -> dict[str, list[str]]`
  - `labels.known_classes(holdout: str | None) -> list[str]`: CLASSES minus that group, order preserved
  - `labels.load_config(path: Path | None = None) -> dict` reads `configs/default.yaml`
  - Fixture `mini_data(tmp_path) -> Path`: a data dir with `mel.npy` float16 shape (120, 128, 64) and `clips.parquet` containing 10 clips per class. Columns: `label, path, MMSI` (0 for background), `hydrophone` (ICLISTENAF2523 / ICLISTENAF2556 alternating for ships, null for background), `mean_distance` (m, null for background and for 1 ship row), `mean_sog` (knots, null for background), `split_f1..split_f5` (vessel-disjoint: grouped by MMSI, ≈80/10/10, every class present in every split), `mel_row = arange`. Each class's mel has a distinct mean offset so a model can learn it.
- `configs/default.yaml` keys and values: `backbone: efficientnet_b0`, `epochs: {class: 25, distance: 20, speed: 20, multi: 25}`, `lr: 3.0e-4`, `weight_decay: 0.01`, `warmup_epochs: 1`, `drop_rate: 0.3`, `patience: 6`, `workers: 4`, `specaugment: {freq_masks: 2, freq_width: 16, time_masks: 2, time_width: 50}`, `norm_clips: 2000`, `max_gpu_temp: 85`, `resume_gpu_temp: 75`, `mc_samples: 20`, `ece_bins: 15`
- `requirements.txt` (torch is NOT listed; the setup scripts install it): timm, numpy, pandas, pyarrow, scikit-learn, scipy, matplotlib, soundfile, tqdm, pyyaml, jupyterlab, ipykernel, nbformat, nbclient, pytest
- `pytest.ini`: `pythonpath = src`, `testpaths = tests`

- [ ] **Step 1: Write failing tests** in `tests/test_labels.py`:
  - `test_classes_fixed_order`: `labels.CLASSES[0] == "background"`, `len == 12`, `labels.CLASSES[-1] == "tug"`
  - `test_groups_disjoint_and_valid`: A/B/C each have 3 classes, are pairwise disjoint, and none is in {background, pilotvessel, dredger}
  - `test_known_classes_holdout_A`: `known_classes("A")` has length 9, keeps CLASSES order, and has no fishing/sailing/tanker. `known_classes(None) == CLASSES`
  - `test_mini_data_fixture`: 120 rows, `mel.shape == (120,128,64)`, and every class appears in train/validation/test of fold 1
- [ ] **Step 2:** Run `python -m pytest tests/test_labels.py -v`. Expected: FAIL (module not found)
- [ ] **Step 3:** Implement `paths.py`, `labels.py`, the configs, `requirements.txt`, `pytest.ini`, `conftest.py`. Install the new test deps: `python -m pip install pyyaml pytest nbformat nbclient`
- [ ] **Step 4:** Run the tests. Expected: 4 passed. Also run `python src/prepare_features.py --help`; it must show defaults under the project root

### Task 2: Data selection, normalisation, dataset

**Files:**
- Create: `src/data.py`, `tests/test_data.py`

**Interfaces:**
- Consumes: `labels.CLASSES`, `labels.known_classes`, `labels.load_groups`
- Produces:
  - `open_features(data_dir: Path) -> tuple[np.memmap, pd.DataFrame]`. Raises `DataError` (subclass of `RuntimeError`) with a message naming the missing or bad file if `mel.npy`/`clips.parquet` are missing, if the row counts differ, or if `mel.npy` is truncated (wrap the `np.load` ValueError).
  - `select_rows(table, *, fold: int, split: str, task: str, holdout: str | None = None, split_mode: str = "vessel", hydrophone_shift: bool = False, seed: int = 0) -> np.ndarray` (sorted row indices)
  - `compute_norm(mel, rows, max_clips=2000, seed=0) -> tuple[np.ndarray, np.ndarray]`: per-band mean/std (float32, shape (128,)); std floored at 1e-3
  - `class MelDataset(torch.utils.data.Dataset)`: `__init__(self, data_dir: Path, table: pd.DataFrame, rows: np.ndarray, known: list[str], norm: tuple, augment: bool, aug_cfg: dict | None = None)`. Opens `mel.npy` lazily per worker (store the path, not the array, so Windows spawn works). `__getitem__` returns dict `x` float32 (1,128,T); `y_class` int64 index in `known` or −1 for an unknown class; `y_dist` float32 km or NaN; `y_speed` float32 knots/10 or NaN; `row` int64
  - `spec_augment(x: torch.Tensor, cfg: dict, generator: torch.Generator | None = None) -> torch.Tensor`: zero-masks (after normalisation) `freq_masks` bands of width ≤ `freq_width` and `time_masks` spans of width ≤ `time_width`
- Selection rules (from spec sections 3 and 5):
  - `vessel`: rows with `split_f{fold} == split`
  - `leaky`: a seeded random permutation of all rows, cut into train/validation/test with **the same counts** as fold `fold`'s vessel split (ignores MMSI on purpose)
  - `holdout` G: for train and validation, drop rows whose label is in group G. Test keeps everything
  - `task in {distance, speed}`: drop background rows and rows with a NaN target
  - `hydrophone_shift`: train/validation keep `hydrophone == "ICLISTENAF2523"` or background; test keeps `hydrophone == "ICLISTENAF2556"` or background

- [ ] **Step 1: Write failing tests** in `tests/test_data.py`, all using `mini_data`:
  - `test_vessel_split_matches_column`: rows equal `table.index[table.split_f1 == "train"]`
  - `test_vessel_split_no_shared_mmsi`: MMSIs > 0 are disjoint across train/test
  - `test_leaky_split_counts_and_overlap`: counts equal the vessel split; with fixture MMSIs (several clips per ship) at least one MMSI is shared between leaky train and test
  - `test_holdout_removes_from_train_keeps_test`: holdout "A" → no fishing/sailing/tanker in train or validation; test still contains them
  - `test_regression_task_drops_background_and_nan`
  - `test_hydrophone_shift`: no `ICLISTENAF2556` in train, no `ICLISTENAF2523` in test
  - `test_norm_shapes_and_floor`
  - `test_dataset_item`: x shape (1,128,64) float32; unknown class → `y_class == -1`; background `y_dist` is NaN; the `y_speed` value equals `mean_sog/10`
  - `test_dataset_in_worker_processes`: a `DataLoader(num_workers=2)` iterates one epoch without error
  - `test_open_features_truncated`: overwriting `mel.npy` with its first 1000 bytes → `DataError` whose message contains `mel.npy`
- [ ] **Step 2:** Run. Expected: FAIL
- [ ] **Step 3:** Implement `src/data.py`
- [ ] **Step 4:** Run. Expected: 10 passed

### Task 3: Device helpers and models

**Files:**
- Create: `src/device.py`, `src/models.py`, `tests/test_device_models.py`

**Interfaces:**
- Produces (`device.py`):
  - `pick_device(pref: str = "auto") -> torch.device`
  - `gpu_memory_gb(device) -> float | None`
  - `auto_batch_size(backbone: str, device: torch.device) -> int`. Table: CPU → 16. MPS → 32. CUDA by memory: <6 GB → effnet 32 / mobilenet 48 / resnet18 48 / tiny 64; 6–11 GB → 64/96/96/128; ≥12 GB → 128/192/192/256
  - `gpu_temperature() -> int | None`: `nvidia-smi --query-gpu=temperature.gpu`; None if missing or erroring
  - `class TemperatureGuard(max_c=85, resume_c=75, check_every=50, poll_s=15)` with `.step() -> float` (seconds paused). It no-ops when `gpu_temperature()` is None
- Produces (`models.py`):
  - `BACKBONES = ("efficientnet_b0", "mobilenetv3_large_100", "resnet18", "tiny_cnn")`
  - `class TinyCNN(nn.Module)`: 4 blocks of conv3×3-BN-ReLU-maxpool2, channels 16/32/64/128, global average pool → 128 features. Attribute `num_features = 128`
  - `class ShipNet(nn.Module)`: `__init__(self, backbone: str, n_classes: int, pretrained: bool = True, drop_rate: float = 0.3)`. timm backbones are created with `in_chans=1, num_classes=0`. `forward(x) -> dict` with keys `logits` (B,n_classes), `distance` (B,), `speed` (B,), `features` (B,D). One dropout before three linear heads
  - `class PretrainedDownloadError(RuntimeError)`: raised by ShipNet when timm cannot fetch weights. Its message includes `--no-pretrained` and "copy the timm cache"
  - `save_checkpoint(path: Path, **state) -> None` (atomic)
  - `load_model(run_dir_or_ckpt: Path, map_location="cpu", which: str = "best") -> tuple[ShipNet, dict]`: returns the model plus a dict with `config`, `known`, `norm`

- [ ] **Step 1: Write failing tests:**
  - `test_device_without_nvidia_smi` (monkeypatch `shutil.which` → None): `gpu_temperature() is None`, `TemperatureGuard().step() == 0.0`, and `auto_batch_size("efficientnet_b0", torch.device("cpu")) == 16`
  - `test_shipnet_tiny_forward`: input (4,1,128,64) → logits (4,12), distance (4,), speed (4,)
  - `test_shipnet_variable_frames`: 501 frames also work
  - `test_checkpoint_roundtrip(tmp_path)`: save → `load_model` → identical outputs in eval mode
  - `test_pretrained_download_failure_message` (monkeypatch `timm.create_model` to raise `OSError` when `pretrained=True`): raises `PretrainedDownloadError` mentioning `--no-pretrained`
- [ ] **Step 2:** Run. Expected: FAIL
- [ ] **Step 3:** Implement
- [ ] **Step 4:** Run. Expected: 5 passed

### Task 4: Metrics and reliability methods

**Files:**
- Create: `src/metrics.py`, `src/reliability.py`, `tests/test_metrics_reliability.py`

**Interfaces:**
- Produces (`metrics.py`, numpy in and out):
  - `accuracy(y, yhat) -> float`, `macro_f1(y, yhat, n_classes) -> float` (classes absent from `y` are skipped), `confusion(y, yhat, n_classes) -> np.ndarray`, `per_class_recall(y, yhat, n_classes) -> np.ndarray` (NaN where absent)
  - `ece(probs, y, n_bins=15) -> float`, `reliability_bins(probs, y, n_bins=15) -> dict(conf, acc, count)`, `nll(probs, y) -> float`, `brier(probs, y) -> float`
  - `open_set_auroc(score, is_unknown) -> float` (higher score = more unknown)
  - `fpr_at_95_tpr(score, is_unknown) -> float`: threshold t = the 95th percentile of `score[~is_unknown]`; return the fraction of unknowns with `score <= t` (unknowns wrongly accepted)
  - `open_set_accuracy(pred_class, score, threshold, y_true_known_idx) -> float`: a clip counts as correct if it's unknown (`y == -1`) and `score > threshold`, or known, `score <= threshold` and `pred == y`
  - `regression_report(y_true, y_pred, bands: list[float]) -> dict(mae, median_ae, rmse, band_accuracy)`
- Produces (`reliability.py`):
  - `softmax(logits, T=1.0)`, `msp_score(logits, T=1.0) = 1 - max softmax`, `energy_score(logits, T=1.0) = -T·logsumexp(logits/T)`
  - `entropy(probs)`, `mutual_information(prob_samples (S,N,C))`
  - `fit_temperature(logits, y) -> float`: minimise NLL over log T with `scipy.optimize.minimize_scalar` (bounded, T in [0.05, 20])
  - `mc_dropout_logits(model, loader, n_samples, device) -> np.ndarray (S,N,C)`: dropout modules in train mode, everything else in eval mode
  - `ensemble_probs(list_of_probs) -> np.ndarray` (mean)

- [ ] **Step 1: Write failing tests** with hand-checkable values:
  - `test_macro_f1_perfect_and_skip_absent`
  - `test_ece_perfectly_calibrated_is_zero`: confidences 0.8, 80 % correct per bin → ECE < 0.01
  - `test_auroc_and_fpr95`: perfectly separated scores → AUROC 1.0, FPR95 0.0. Identical scores → AUROC 0.5
  - `test_open_set_accuracy_counts`: a 4-sample hand case → 0.75
  - `test_regression_report_bands`
  - `test_energy_lower_for_confident`: logits [10,0,0] have lower energy than [1,1,1]
  - `test_fit_temperature_recovers_overconfidence`: logits = 3 × well-calibrated logits → fitted T ≈ 3 (±0.3)
  - `test_mc_dropout_varies`: a TinyCNN ShipNet with drop 0.5 gives S=5 samples that are not all equal, and the BN running stats are unchanged
- [ ] **Step 2:** Run. Expected: FAIL
- [ ] **Step 3:** Implement
- [ ] **Step 4:** Run. Expected: 8 passed

### Task 5: Training CLI

**Files:**
- Create: `src/train.py`, `tests/test_train.py`

**Interfaces:**
- Consumes: everything from Tasks 1–4
- Produces:
  - `make_parser() -> argparse.ArgumentParser`, `run(args) -> Path` (the run dir)
  - CLI flags: `--task {class,distance,speed,multi}` (required), `--fold 1..5` (default 1), `--backbone`, `--holdout {A,B,C}`, `--split-mode {vessel,leaky}`, `--hydrophone-shift`, `--seed` (0), `--epochs` (config per task), `--batch-size` (0 = auto), `--lr`, `--weight-decay`, `--workers`, `--init-from PATH` (backbone weights from another run's `best.pt`; heads skipped if shapes differ), `--no-pretrained`, `--resume`, `--max-hours FLOAT`, `--max-gpu-temp`, `--limit-train N`, `--limit-val N`, `--name`, `--data-dir`, `--runs-dir`, `--device`
  - Default run name: `{task}_f{fold}_{backbone}_{split_mode}` + `_h{G}` + `_hyd` + `_s{seed}` when not default
  - Run dir contents: `config.json` (all resolved args + known classes + git-free code version = sha256 of `src/*.py`), `norm.npz`, `last.pt`, `best.pt`, `log.csv` (epoch, train_loss, val_loss, val_metric, lr, seconds, clips_per_s, gpu_temp_max, paused_s), `done.json` (written when training finishes normally or by early stopping)
  - Losses: class → cross-entropy with inverse-frequency class weights from the train rows. distance/speed → Huber δ 0.2 on the matching head, masked to finite targets. multi → CE + Huber(distance) + Huber(speed)
  - Validation metric: class/multi → macro-F1 (higher is better). distance → MAE in m. speed → MAE in knots (lower is better)
  - Optimiser AdamW; linear warmup for `warmup_epochs`, then cosine decay to 0; AMP + GradScaler on CUDA only; early stopping on `patience`; a `TemperatureGuard` step every batch; `keep_awake()` around `run`
  - `--resume` restores model, optimiser, scheduler, scaler, epoch, best metric, patience counter and RNG states from `last.pt`. With an existing run dir and no `--resume` → exit with a message (never silently overwrite)
  - `--max-hours`: after any epoch, if elapsed > the limit, save and exit with "stopped for time; resume with --resume"

- [ ] **Step 1: Write failing tests** (all with `--backbone tiny_cnn --no-pretrained --workers 0` on `mini_data`):
  - `test_train_class_writes_run`: 2 epochs → `log.csv` has 2 rows, and `best.pt`, `norm.npz`, `done.json` exist
  - `test_train_learns_mini`: 15 epochs → val macro-F1 > 0.5 (the fixture's classes are separable)
  - `test_train_distance_ignores_background`: the run's selected train rows contain no background (read from `config.json` → `n_train`, matched against `select_rows`)
  - `test_holdout_known_classes`: `config.json["known"]` has 9 classes for `--holdout A`
  - `test_resume_after_kill`: run 3 epochs with `--max-hours 0` (stops after epoch 1); then leave a garbage `last.pt.tmp` behind (simulating a kill mid-write); `--resume --epochs 3` → `log.csv` has 3 rows with epochs 1,2,3 and no duplicates
  - `test_existing_run_requires_resume`: a second run with the same name and no `--resume` → `SystemExit` with a message containing `--resume`
  - `test_init_from_class_run`: a distance run with `--init-from <class run>/best.pt` loads backbone weights (the first conv weight equals the source)
- [ ] **Step 2:** Run. Expected: FAIL
- [ ] **Step 3:** Implement
- [ ] **Step 4:** Run. Expected: 7 passed

### Task 6: Evaluation CLI and figures

**Files:**
- Create: `src/evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `models.load_model`, `data.*`, `metrics.*`, `reliability.*`, `features.LogMel`, `shipzip.SplitZip` (noise test only)
- Produces:
  - `evaluate_run(run_dir: Path, *, ensemble: list[Path] = (), mc_samples: int = 0, noise_snrs: tuple = (), audio_zip: Path | None = None, noise_limit: int = 1000, split: str = "test") -> dict`
  - Writes `run_dir/eval/`: `metrics.json`, `predictions.parquet` (row, label, logits_*, msp, energy, [mc_entropy, mc_mi], [ens_entropy], pred_distance_m, pred_speed_kn), `calibration.json` (`temperature`, `msp_threshold`, `energy_threshold`, chosen on **validation** so that 95 % of known validation clips are accepted), and figures `confusion.png`, `reliability_before_after.png`, `accuracy_by_distance.png`, plus `open_set_roc.png` (if holdout), `distance_scatter.png` / `speed_scatter.png` (if regression), `noise_curve.png` (if noise)
  - `metrics.json` sections:
    - `closed_set`: accuracy, macro_f1, per_class_recall dict, n_test, `rare_class_warning` listing classes with < 5 test ships
    - `calibration`: ECE, NLL and Brier before and after temperature
    - `open_set` (holdout runs): AUROC and FPR95 for msp, energy, mc_entropy, mc_mi, ens_entropy, plus open_set_accuracy at the validation thresholds
    - `by_distance`: accuracy per band
    - `regression`: the report plus the "predict train median" baseline MAE
    - `noise`: accuracy, and the fraction flagged unknown, per SNR
  - Noise test: for up to `noise_limit` test clips, read the WAV from the zip, add white Gaussian noise at the given SNR (relative to the clip's RMS), recompute log-mel with `features.LogMel`, normalise with the run's `norm.npz`, predict
  - CLI: `python src/evaluate.py RUN_DIR [--ensemble RUN_DIR ...] [--mc 20] [--noise 20 10 0] [--audio-zip PATH]`

- [ ] **Step 1: Write failing tests** (tiny runs trained in a module-scoped fixture on `mini_data`):
  - `test_evaluate_closed_set_outputs`: the metrics keys exist, `confusion.png` exists, and `calibration.json["temperature"] > 0`
  - `test_thresholds_from_validation_not_test`: monkeypatch the test-split loader to return shuffled labels → `calibration.json` is unchanged
  - `test_evaluate_holdout_open_set`: a holdout-A run → the `open_set` section has msp and energy AUROC in [0,1]
  - `test_evaluate_ensemble_and_mc`: 2 runs + `mc_samples=3` → `ens_entropy` and `mc_mi` columns exist
  - `test_evaluate_regression`: a distance run → MAE is in metres (> 1) and the baseline MAE is present
- [ ] **Step 2:** Run. Expected: FAIL
- [ ] **Step 3:** Implement
- [ ] **Step 4:** Run. Expected: 5 passed

### Task 7: Classical baseline

**Files:**
- Create: `src/baseline_classical.py`, `tests/test_baseline.py`

**Interfaces:**
- Produces: `band_stats(mel_block: np.ndarray) -> np.ndarray` returning (N, 257): per-band mean (128), per-band std (128) and the overall mean dB (1). `run_baseline(data_dir, fold, out_dir, max_train=20000, seed=0) -> dict`, which writes `runs/baseline_f{fold}/metrics.json` with class (accuracy, macro-F1, per-class recall), distance (MAE m) and speed (MAE kn). Models: `HistGradientBoostingClassifier(class_weight="balanced")` and `HistGradientBoostingRegressor(loss="absolute_error")`. Train on the fold train rows (a seeded subsample if larger than `max_train`), evaluate on test. CLI: `python src/baseline_classical.py --fold 1`
- [ ] **Step 1: Failing tests:** `test_band_stats_shape`; `test_baseline_runs_on_mini` (metrics.json exists and accuracy > 1/12)
- [ ] **Step 2:** Run. Expected: FAIL
- [ ] **Step 3:** Implement
- [ ] **Step 4:** Run. Expected: 2 passed

### Task 8: Prediction on new audio

**Files:**
- Create: `src/predict.py`, `tests/test_predict.py`

**Interfaces:**
- Produces: `predict_file(wav_path: Path, class_run: Path, distance_run: Path | None = None, speed_run: Path | None = None, score: str = "energy") -> pd.DataFrame` with columns `start_s, end_s, class, confidence, unknown (bool), distance_m, speed_kn`, plus a final row `start_s = "overall"`: the majority known class, mean confidence, median distance and speed over ship windows.
- Behaviour:
  - Decode with `features.decode_wav` (downmixes to mono), then `features.resample` to 32 kHz.
  - Cut into 5 s windows with a 5 s hop. A final partial window of ≥ 2.5 s is padded with `features.fix_length`; shorter is dropped. A file under 2.5 s → `ValueError("need at least 2.5 s of audio")`.
  - Temperature and thresholds come from `class_run/eval/calibration.json`. If it's missing: T = 1, MSP threshold 0.5, and a printed warning to run `evaluate.py` first.
  - "Unknown" if the score exceeds the threshold. Distance and speed are blank for background or unknown windows.
  - The CLI prints the table and a one-line reminder that distance is calibrated only for ONC icListen hydrophones.
  - CLI: `python src/predict.py WAV --class-run RUN [--distance-run RUN] [--speed-run RUN] [--score energy|msp]`
- [ ] **Step 1: Failing tests:** `test_predict_odd_wavs` (synthetic 48 kHz stereo 12 s → 2 full windows + 1 padded = 3 window rows + overall; 3 s mono 16 kHz → 1 row + overall; 2 s → ValueError); `test_predict_without_calibration_warns` (`capsys` contains "evaluate.py")
- [ ] **Step 2:** Run. Expected: FAIL
- [ ] **Step 3:** Implement
- [ ] **Step 4:** Run. Expected: 2 passed

### Task 9: Portability: check_setup, benchmark, setup scripts, README

**Files:**
- Create: `src/check_setup.py`, `src/benchmark.py`, `setup/setup_windows.ps1`, `setup/setup_unix.sh`, `README.md`, `tests/test_portability.py`

**Interfaces:**
- `check_setup.main(argv=None) -> int` (0 = OK). Prints the Python and torch versions, device, GPU name and memory, and the data files. Verifies `data/checksums.json` (`{"clips.parquet": sha256, "mel.npy": {"size": int, "sha256_first_64MB": str, "sha256_last_64MB": str}}`). `--write` creates that file (run once on this machine). Failures print exactly which file to copy again.
- `benchmark.main(argv=None) -> dict`: trains `--steps 30` batches and evaluates 30 batches with the chosen backbone and auto batch size, then prints clips/s and estimated hours for each experiment row in spec section 5. Runs per experiment: E2 1, E3 3, E5 1, E7 1, E9 1, E10 1, ensemble +2. Epochs come from `default.yaml`; the estimate assumes 80 % of the max epochs because of early stopping. Returns the dict that was printed.
- Setup scripts:
  - Create `.venv`.
  - If `nvidia-smi` exists, `pip install torch --index-url https://download.pytorch.org/whl/cu128`; otherwise plain `pip install torch` (CPU / MPS).
  - `pip install -r requirements.txt`, then `python src/check_setup.py`.
  - Neither script needs admin rights.
- `README.md`: what to copy to the friend's laptop (spec section 11), setup, check, benchmark, then the commands in spec section 6 order (baseline → E2 → evaluate → E3 groups A/B/C → ensemble seeds 1–2 → E9 distance with `--init-from` → E10 speed → E5 leaky → E7 hydrophone shift → evaluate each, with `--noise` if the zip is present), stopping/resuming, Kaggle notes, and the lesson list.
- [ ] **Step 1: Failing tests:** `test_check_setup_ok` (mini_data + `--write`, then verify → 0); `test_check_setup_truncated_mel` (truncate after write → non-zero, output mentions `mel.npy`); `test_benchmark_returns_estimates` (tiny_cnn, `--steps 3` on mini_data → a dict with an `E2` key and a positive number of hours)
- [ ] **Step 2:** Run. Expected: FAIL
- [ ] **Step 3:** Implement
- [ ] **Step 4:** Run. Expected: 3 passed

### Task 10: Real-data smoke test on this laptop (not training)

**Files:**
- Create: `data/checksums.json` (via `check_setup.py --write`)
- Modify: `README.md` (paste the measured benchmark numbers for the RTX 3050 as a reference row)

- [ ] **Step 1:** `python src/check_setup.py --write`, then `python src/check_setup.py`. Expected: exit 0, CUDA listed
- [ ] **Step 2:** `python src/train.py --task class --limit-train 512 --limit-val 256 --epochs 1 --name smoke_class --workers 2`. Expected: the run dir exists, `log.csv` has 1 row, and pretrained EfficientNet-B0 weights were downloaded
- [ ] **Step 3:** `python src/evaluate.py runs/smoke_class`. Expected: `eval/metrics.json` and the figures exist (scores near chance are fine)
- [ ] **Step 4:** `python src/predict.py <one WAV extracted from the zip to scratchpad> --class-run runs/smoke_class`. Expected: a table prints
- [ ] **Step 5:** `python src/benchmark.py`. Expected: estimates print; record them in the README
- [ ] **Step 6:** Delete `runs/smoke_*`, so the friend's laptop starts clean

### Task 11: Lessons 00–04 and notebook builder

**Files:**
- Create: `notebooks/build_notebooks.py`, `notebooks/lessons/00_setup_check.py`, `01_explore_data.py`, `02_sound_to_spectrogram.py`, `03_classical_baseline.py`, `04_tiny_cnn_from_scratch.py`, the matching `notebooks/*.ipynb` (generated), and `tests/test_notebooks.py`

**Interfaces:**
- `build_notebooks.py`: converts each `# %%` / `# %% [markdown]` file in `notebooks/lessons/` to `notebooks/<same name>.ipynb` with nbformat; `--execute NAME` runs one through nbclient (timeout 900 s)
- Lesson style: plain-language markdown first (what and why), then short code that imports from `src/` and runs on the real `data/`. Each lesson ends with 2–3 "try it" exercises. Content per spec section 8:
  - **00:** device, GPU memory, data checksums
  - **01:** CSV/table columns, class/ship counts, the rare-class table, fold sizes, playing 3 clips with `IPython.display.Audio` (read from the zip if present; otherwise skipped with a note), why vessel-disjoint matters
  - **02:** a waveform → FFT → STFT → mel filterbank plot → dB, built step by step with `features.py`, plus how changing `N_FFT`/`HOP` changes the image
  - **03:** `band_stats`, train/val/test, accuracy vs macro-F1, a confusion matrix, and the algorithm-selection discussion
  - **04:** convolution by hand on one spectrogram, then a TinyCNN trained 3 epochs on 2,000 clips with live train/val curves showing overfitting
- [ ] **Step 1: Failing test:** `test_build_creates_valid_notebooks` (all 5 `.ipynb` validate with `nbformat.validate`)
- [ ] **Step 2:** Run. Expected: FAIL
- [ ] **Step 3:** Write the builder and lessons 00–04
- [ ] **Step 4:** Run the test. Expected: PASS. Then `python notebooks/build_notebooks.py --execute 00_setup_check` and `--execute 02_sound_to_spectrogram`. Expected: both complete without errors (they use real data, CPU or GPU)

### Task 12: Lessons 05–10

**Files:**
- Create: `notebooks/lessons/05_training_the_real_model.py`, `06_tuning_and_model_selection.py`, `07_unknown_ship_detection.py`, `08_trustworthy_confidence.py`, `09_distance_and_speed.py`, `10_stress_tests_and_report.py`, plus the generated `.ipynb`
- Modify: `tests/test_notebooks.py` (expect 11 notebooks)

**Interfaces:**
- Each lesson explains its concepts (spec section 8 rows 05–10), shows the exact `train.py` / `evaluate.py` commands, then **reads finished runs from `runs/` if they exist**, and otherwise prints "no run yet, run: …" and skips the result cells (guarded by `if run_dir.exists()`).
- **05:** loss, optimiser, LR schedule, batch size, augmentation, AMP, checkpoints, early stopping; plots `log.csv` curves.
- **06:** the order to tune in (LR, then augmentation, then backbone), an LR range test on 300 batches (`torch` loop inline), sweeps only on validation, a comparison table of backbones.
- **07:** reads `eval/metrics.json` of the holdout runs and builds the AUROC/FPR95 table plus ROC plots.
- **08:** reliability diagrams before and after temperature, ECE, MC dropout vs ensemble.
- **09:** distance and speed scatter, errors by band, why loudness drives distance.
- **10:** vessel vs leaky gap, accuracy by distance, hydrophone shift, the noise curve, and assembling the final report figures into `runs/report/`.
- [ ] **Step 1:** Update the test to expect 11 notebooks. Run. Expected: FAIL
- [ ] **Step 2:** Write the lessons 05–10
- [ ] **Step 3:** Run the test. Expected: PASS. Run `--execute` on 05 and 07 with no runs present. Expected: both complete and print "no run yet"
- [ ] **Step 4:** Run the full suite: `python -m pytest -v`. Expected: all pass
