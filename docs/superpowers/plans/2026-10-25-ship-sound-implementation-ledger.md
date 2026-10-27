# SDD ledger — plan: D:\shipsound\docs\superpowers\plans\2026-10-25-ship-sound-implementation.md

Spec: D:\shipsound\docs\2026-10-24-ship-sound-plan.md (v3) — read.
Setup: Ruling: no git repo (user never asked for one; commits only on request) → no worktree, no sdd scripts, manual ledger here; BASE = n/a — cost if wrong: no per-task diff ranges for the final review; review reads files directly.
Setup: Ruling: NVIDIA GPU is disabled in Device Manager (code 22, done by the separate system-diagnosis session) → all smoke runs on CPU; do not re-enable — cost if wrong: slower smoke tests only.

## Pre-flight (shared interfaces)
| Producer → Consumer | Produces vs consumes | Finding |
|---|---|---|
| T1 → T2,T5,T6 | labels.CLASSES / known_classes / load_groups / load_config | consistent |
| T2 → T5,T6,T7,T9 | select_rows, compute_norm, MelDataset, open_features | consistent |
| T3 → T5,T6,T8 | save_checkpoint(**state) vs load_model → (ShipNet, {config, known, norm}) | plan leaves checkpoint keys open. Ruling: checkpoint dict = {model, config(backbone, drop_rate, task, ...), known, norm{mean,std}, epoch, best_metric, optimizer, scheduler, scaler, patience_left, rng}; load_model rebuilds ShipNet(config.backbone, len(known), pretrained=False, drop_rate) — cost if wrong: one key rename |
| T4 → T6 | metrics/reliability function names | consistent |
| T6 → T8 | calibration.json {temperature, msp_threshold, energy_threshold} | consistent |
| T5 → T9 | benchmark needs model+loader construction | Ruling: benchmark builds ShipNet + MelDataset directly (no train.py internals) — cost if wrong: small duplication |

## Tasks
Task 1: Ruling: added test_load_config_defaults (plan listed 4 tests; load_config is in Produces) — cost if wrong: none.
Task 1: Ruling: prepare_features --help does not print defaults (RawDescription formatter); verified paths.DATA_DIR/AUDIO_ZIP resolve under project root instead — cost if wrong: none.
Task 1: complete (tests: python -m pytest -q → 5 passed)
Task 2: complete (tests: python -m pytest -q → 17 passed; real data fold1 counts checked)
Task 3: Ruling: torch.load uses weights_only=True (security hook); therefore checkpoints must contain only tensors + plain dict/list/str/int/float (no numpy arrays, no Path, numpy RNG stored as plain values) — applies to Task 5 — cost if wrong: resume fails to load, caught by test_resume_after_kill.
Task 3: Ruling: added tests test_auto_batch_size_cuda_table, test_shipnet_timm_backbone_offline, test_unknown_backbone_rejected beyond the plan's 5 — cost if wrong: none.
Task 3: complete (tests: python -m pytest -q → 25 passed)
Task 4: complete (tests: python -m pytest -q → 35 passed; 10 metric/reliability tests incl. extra nll/brier and entropy/MI checks)
Task 5: Ruling: --init-from copies every weight whose name+shape match (not only backbone) and returns skipped names via train.load_init_weights(model, path); tested directly instead of through a full distance run — cost if wrong: an untrained head copied over, harmless.
Task 5: Ruling: checkpoint RNG = torch + python + DataLoader generator (+cuda); numpy global RNG not saved (training never uses it after seeding; keeps weights_only=True loading) — cost if wrong: augmentation randomness differs slightly after resume.
Task 5: Ruling: added test_train_multi_task_runs — cost if wrong: none.
Task 5: complete (tests: python -m pytest -q → 43 passed)
Task 6: Ruling: metrics.json closed-set/calibration computed on KNOWN test clips only (unknown-class clips have no correct class); open-set section covers all test clips — cost if wrong: none (spec-consistent).
Task 6: Ruling: ensemble members averaged at T=1; MC-dropout/ensemble get AUROC/FPR95 but open-set accuracy only for msp/energy (their thresholds need validation passes not in the plan) — cost if wrong: one more validation pass to add later.
Task 6: Ruling: noise test is not unit-tested (needs the 16.6 GB zip); exercised in Task 10 smoke on real data — cost if wrong: noise bug found later.
Task 6: complete (tests: python -m pytest -q → 49 passed)
Task 7: complete (tests: python -m pytest -q → 51 passed)
Task 8: Ruling: plan's "12 s → 2 full + 1 padded" contradicts its own ≥2.5 s rule (12 s leaves a 2 s tail, dropped); test uses 13 s → 3 windows — cost if wrong: none.
Task 8: Ruling: without calibration.json the script falls back to MSP 0.5 even if --score energy (no energy threshold exists yet) and prints the evaluate.py hint — cost if wrong: none.
Task 8: Ruling: added test_predict_uses_calibration_and_blanks_unknown — cost if wrong: none.
Task 8: complete (tests: python -m pytest -q → 54 passed)
Task 9: Ruling: benchmark returns unrounded hours (rounding to 2 decimals zeroed small workloads; found by test) — cost if wrong: none.
Task 9: Ruling: added test_check_setup_changed_parquet; setup_windows.ps1 verified by an actual run here (exit 0, torch stayed cu128); setup_unix.sh syntax-checked only (no Linux/mac here) — cost if wrong: unix script bug found on first friend run.
Task 9: Ruling: setup scripts install torch AND torchvision from the CUDA index first (timm needs torchvision; installing it later from PyPI tried to swap in a CPU torch earlier in this project) — cost if wrong: none.
Task 9: complete (tests: python -m pytest -q → 58 passed; setup_windows.ps1 exit 0)
Task 10: Ruling: GPU disabled → smoke ran on CPU; README reference speed records CPU numbers (14 train / 63 eval clips/s, ≈117 h total) with a note that a GPU machine/Kaggle is needed — cost if wrong: none.
Task 10: Ruling: models.py sets HF_HUB_DISABLE_SYMLINKS_WARNING=1 (noisy Windows warning seen in smoke run would alarm a beginner) — cost if wrong: none.
Task 10: extra check: stored mel vs mel recomputed from zip audio differ ≤ 0.08 dB (float16 rounding) → predict/noise paths match training features.
Task 10: complete (checksums written+verified; smoke train 1 epoch/512 clips OK; evaluate incl. --mc 2 --noise 10 0 on real test set OK, figures render; predict on real WAV OK; benchmark OK; runs/ deleted; pytest → 58 passed)
Task 11: Ruling: lesson 02 originally claimed "farther ships are quieter"; real data shows overall loudness vs distance is weak (Spearman -0.10; best band -0.36 across all ships) but strong within a class (cargo -0.82, tanker -0.84) → lesson rewritten to show both; supports --init-from for distance — cost if wrong: none (data-backed).
Task 11: Finding: fold 1 validation split is 37.8 % dredger (few ships). Ruling: temperature + unknown thresholds now fitted with class-balanced weights (evaluate.calibrate, metrics.class_balanced_weights, metrics.weighted_percentile, reliability.fit_temperature(weights=)) — tests test_fit_temperature_weights, test_weighted_percentile, test_class_balanced_weights, test_calibrate_invariant_to_class_duplication RED→GREEN — cost if wrong: thresholds differ from an unweighted fit; spec only says "validation", balanced is the faithful reading given the skew.
Task 11: Ruling: lesson 04 overfitting demo changed to 48 balanced clips / 60 epochs / lr 3e-3 (plan's 200 clips/15 epochs reached only 36 % train acc on real data, so it didn't show overfitting); validation subsets class-balanced; lesson 01 + 08 explain the dredger skew — cost if wrong: none.
Task 11: complete (tests: pytest tests/test_notebooks.py → 2 passed; lessons 00-04 executed on real data without errors: 03 → random forest best on validation, test acc 0.315 / macro-F1 0.278)
Task 12: all 11 lessons built and executed on real data (CPU): 0 errors; 05/07/08/10 print the "no run yet" commands; 06 LR range test (tiny_cnn, 100 steps) → lowest loss at 3.4e-3, i.e. ~3e-4 recommended = config default; 04 overfitting demo shows train acc ~0.75-0.79 vs val ~0.2, val loss rising (text adjusted to match).
Task 12: complete (tests: pytest tests/test_notebooks.py → 2 passed; full execution of 11 notebooks → 0 errors)
Final review: fresh reviewer (opus) — verdict "with fixes": 1 Critical, 5 Important, ~17 Minor.
Final: Re-grade: promoted to Important by effect on the user: temperature guard ineffective below 75 °C (safety feature the user explicitly cares about), log.csv locked by Excel crashes training (likely on Windows), --init-from/--ensemble across folds leaks test clips, --noise without zip fails after the whole evaluation, no device smoke test (unknown friend GPU), lesson 06 OOM on 4 GB + ranks leaky/holdout runs as model-selection candidates, README false claims (keep-awake on mac/linux, Kaggle resume path, torch minimum), resume of a finished run crashes (UnboundLocalError).
Final: fixed resume crash on GPU (map_location=device put RNG states on GPU) — test_resume_loads_checkpoint_on_cpu RED→GREEN (test tightened: it passed on CPU before tightening); also scaler state only loaded when non-empty (CPU→GPU resume) — not testable without CUDA.
Final: fixed raw tracebacks — PretrainedDownloadError raised "from None"; train/evaluate/benchmark/baseline main() exit with one-line "error: ..." — test_pretrained_error_hides_http_traceback, test_main_missing_data_is_plain_message, test_benchmark_and_baseline_missing_data_messages RED→GREEN.
Final: fixed evaluate overwriting earlier sections — merge with previous metrics/predictions when best.pt fingerprint + split match — test_evaluate_keeps_earlier_sections, test_evaluate_drops_sections_from_old_model RED→GREEN.
Final: fixed ensemble experiment unable to answer E3 — README/lessons 07/08/10 + benchmark.EXPERIMENTS use group-A seeds (…_hA_s1/_s2) — test_benchmark_counts_holdout_ensemble RED→GREEN.
Final: fixed predict.py whole-file batch — CHUNK=32 windows, log-mel computed once per chunk for all heads — test_predict_chunked_matches_unchunked RED→GREEN.
Final: fixed E7 confound — lesson 10 compares E2 vs E7 on the same ICLISTENAF2556 test clips (doc/lesson change, verified by stand-in run execution).
Final: fixed temperature guard below 75 °C (resume_c = min(resume, max-10)) — test_temperature_guard_low_limit_still_pauses RED→GREEN (test's off-by-one corrected: guard waits until strictly below resume temp, as its message says).
Final: fixed log.csv lock — train.append_log keeps rows pending on PermissionError and retries — test_log_write_survives_locked_file RED→GREEN.
Final: fixed --init-from / --ensemble across fold/split/holdout/hydrophone — test_init_from_refuses_other_fold, test_ensemble_refuses_other_fold RED→GREEN.
Final: fixed --noise zip check up front — test_noise_without_zip_fails_before_work RED→GREEN.
Final: fixed resume of finished run — test_resume_finished_run_message RED→GREEN.
Final: fixed no device smoke test — check_setup.device_smoke_test + README "GPU problems" — test_check_setup_device_failure_message RED→GREEN.
Final: fixed lesson 06 (LR test with AMP + half batch; runs_table shows split/holdout/hyd and ranks only comparable runs) and README (keep-awake per OS, Kaggle resume copy step, torch ≥ 2.3) — docs, verified by notebook execution.
Final: minor (deferred): --resume does not compare arguments with the saved config.json.
Final: minor (deferred): --init-from still downloads ImageNet weights first (harmless once cached).
Final: minor (deferred): MC dropout reruns the whole backbone 20x (only the head dropout is random) — ~20x slower than needed.
Final: minor (deferred): spec §8 lesson 05 mentions mixup; not implemented or explained.
Final: minor (deferred): last.pt is saved before the log row; gpu_temp_max in log.csv is the session maximum, not per epoch.
Final: minor (deferred): PretrainedDownloadError wraps every exception from timm.create_model, not only network errors.
Final: minor (deferred): requirements.txt unpinned; setup scripts hard-code the cu128 index (mitigated by the device smoke test + README GPU section).
Final: stand-in integration check — 8 tiny runs under the real run names trained + evaluated (incl. --mc then --noise merge, holdout ensemble) and lessons 05-10 executed against them: 0 errors; stand-ins deleted afterwards.
Final: last full notebook execution was stopped by Claude Code for low system memory after lessons 00-02; all 11 notebooks rebuilt from final sources and valid; 03-10 have no saved outputs (each ran without errors earlier in this session). Not restarted (harness instruction).
Final: suite 79/79 passed after the fix pass. Not a git repo → superpowers:finishing-a-development-branch not applicable; ledger kept as docs/superpowers/plans/2026-10-25-ship-sound-implementation-ledger.md.
