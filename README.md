# Reliability-Aware Underwater Vessel Recognition

Recognise the ship class from a 5-second hydrophone clip **on ships the model has never heard**, say
**"unknown"** when it isn't sure, give a confidence you can trust, and estimate the ship's **distance** and **speed**.
Data: the [UniqueShip](https://uniqueshipdata.org/) 12-class / 5-hour subset, recorded by
[Ocean Networks Canada](https://data.oceannetworks.ca/). The data is not in this repository; see the UniqueShip
website for access and for how to cite it.

- Design and decisions: `docs/2026-10-24-ship-sound-plan.md`
- Lessons (learn every step): `notebooks/` (start JupyterLab with `jupyter lab notebooks`)

---

## 1. Copy the project to the training laptop

Copy these (≈ 5.6 GB) by USB drive or Google Drive, keeping the folder structure:

| Copy | Size | Needed for |
|---|---|---|
| `src/ configs/ notebooks/ docs/ setup/ tests/ README.md requirements.txt pytest.ini` | < 5 MB | everything |
| `data/` (`mel.npy`, `clips.parquet`, `features.json`, `checksums.json`) | 5.5 GB | everything |
| `gdrive_download/audio/` (`allclass_5hours.z01` + `.zip`) | 16.6 GB | **optional**: noise test (E8), listening to clips in lesson 01 |

Do **not** copy `.venv/` (it is machine-specific) or `runs/`.

## 2. Set up (once)

Windows:
```
powershell -ExecutionPolicy Bypass -File setup\setup_windows.ps1
.venv\Scripts\activate
```
Linux / macOS:
```
bash setup/setup_unix.sh
source .venv/bin/activate
```
The script installs the right PyTorch for the GPU, then runs `python src/check_setup.py`, which must end with
**OK - ready to train**. If it names a damaged file, copy that file again.

The first training run downloads ImageNet weights (~20 MB) from the internet. Offline? Add `--no-pretrained`.

## 3. How long will it take on this machine?

```
python src/benchmark.py
```
About one minute. It prints the estimated hours for every experiment below.

## 4. Experiments, in order

Every command saves progress after each epoch. Stop any time with **Ctrl+C**, then repeat the same command
with `--resume`. Add `--max-hours 1.5` to train in sessions. Results go to `runs/<run name>/`.
Keep the laptop **plugged in with the lid open**. On Windows the scripts stop it from sleeping by themselves; on macOS
start them as `caffeinate -i python src/train.py ...`, on Linux as `systemd-inhibit python src/train.py ...`.

| # | What | Commands |
|---|---|---|
| E1 | Classical baseline (CPU, minutes) | `python src/baseline_classical.py --fold 1` |
| E2 | 12-class CNN | `python src/train.py --task class` then `python src/evaluate.py runs/class_f1_efficientnet_b0_vessel --mc 20` |
| E3 | Unknown ships, groups A, B, C | `python src/train.py --task class --holdout A` then `python src/evaluate.py runs/class_f1_efficientnet_b0_vessel_hA --mc 20` (repeat for B and C) |
| E3b | Deep ensemble for unknown ships (optional; good for Kaggle) | `python src/train.py --task class --holdout A --seed 1` and `--seed 2`, then `python src/evaluate.py runs/class_f1_efficientnet_b0_vessel_hA --mc 20 --ensemble runs/class_f1_efficientnet_b0_vessel_hA_s1 runs/class_f1_efficientnet_b0_vessel_hA_s2` |
| E9 | Distance | `python src/train.py --task distance --init-from runs/class_f1_efficientnet_b0_vessel/best.pt` then `python src/evaluate.py runs/distance_f1_efficientnet_b0_vessel` |
| E10 | Speed | `python src/train.py --task speed --init-from runs/class_f1_efficientnet_b0_vessel/best.pt` then `python src/evaluate.py runs/speed_f1_efficientnet_b0_vessel` |
| E5 | Leaky split (shows inflated accuracy) | `python src/train.py --task class --split-mode leaky` then `python src/evaluate.py runs/class_f1_efficientnet_b0_leaky` |
| E7 | Different hydrophone | `python src/train.py --task class --hydrophone-shift` then `python src/evaluate.py runs/class_f1_efficientnet_b0_vessel_hyd` |
| E8 | Added noise (needs the audio zip) | `python src/evaluate.py runs/class_f1_efficientnet_b0_vessel --noise 20 10 0` |

Run folder names follow `{task}_f{fold}_{backbone}_{vessel|leaky}`, plus `_h{group}`, `_hyd` or `_s{seed}`.

E4 (calibration) and E6 (accuracy by distance) come out of every class evaluation automatically.
Each evaluation writes `runs/<run>/eval/metrics.json` plus figures (confusion matrix, reliability diagram,
ROC curves, scatter plots). Evaluating the same run again (e.g. E8 after E2) **adds** to the earlier results;
they are only replaced if the model was retrained.

Useful options for `train.py`: `--fold 2..5` (other folds), `--backbone mobilenetv3_large_100 | resnet18 | tiny_cnn`,
`--batch-size N` (if "out of memory"), `--epochs N`, `--max-hours H`, `--resume`.

## 5. Predict a new recording

```
python src/predict.py recording.wav --class-run runs/class_f1_efficientnet_b0_vessel \
    --distance-run runs/distance_f1_efficientnet_b0_vessel --speed-run runs/speed_f1_efficientnet_b0_vessel
```
Any sample rate, mono or stereo, at least 2.5 s. Distance is only trustworthy for recordings from the same kind
of hydrophone (ONC icListen) at the same gain.

## 6. Training on Kaggle instead (free GPU, keeps running with the browser closed)

1. kaggle.com → Datasets → New dataset → upload the `data/` folder (private).
2. New notebook → Add data → your dataset; Settings → Accelerator: GPU; Internet: on.
3. Upload this project's `src/`, `configs/` and `requirements.txt` as a second dataset (or paste them), then in
   the notebook:
   ```
   !pip install -q timm pyyaml
   import os; os.environ["SHIPSOUND_DATA"] = "/kaggle/input/<your-data-dataset>"
   !python /kaggle/input/<your-code-dataset>/src/train.py --task class --runs-dir /kaggle/working/runs --max-hours 11.5
   ```
4. "Save version → Save & Run All" runs in the background (max ~12 h per session, ~30 GPU h per week).
   Download `/kaggle/working/runs` afterwards. To continue a run in a new session, upload the run folder as a
   dataset, copy it to the writable folder first (`!mkdir -p /kaggle/working/runs && cp -r /kaggle/input/<runs-dataset>/<run> /kaggle/working/runs/`),
   then run the same command with `--resume`.

## GPU problems

- Needs **PyTorch 2.3 or newer** (`python -c "import torch; print(torch.__version__)"`).
- `check_setup.py` runs a tiny calculation on the GPU. If it prints `PROBLEM: PyTorch cannot compute on cuda`
  (often "no kernel image is available"), the installed PyTorch build doesn't support that (older) GPU. Reinstall
  a build for an older CUDA version, e.g.
  `python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126`
  (or `.../whl/cu118`), then run `check_setup.py` again.
- "CUDA out of memory" during training: add `--batch-size 16` (or 8).

## 7. Tests

```
python -m pytest
```
Runs in about a minute on a small synthetic dataset (no GPU needed).

## Reference speed

Measured with `python src/benchmark.py` on the preparation laptop (i5-12500H, **CPU only**: its GPU was switched off
at the time): training 14 clips/s, evaluation 63 clips/s → E2 ≈ 14 h, all experiments ≈ 117 h.
**CPU-only training is not practical**: use a machine with an NVIDIA GPU or Kaggle. Run the benchmark there for
real numbers (section 3).
