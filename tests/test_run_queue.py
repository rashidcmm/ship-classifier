import run_queue

E2 = "class_f1_efficientnet_b0_vessel"


def ids(steps):
    return [s for s, _ in steps]


def test_queue_follows_readme_order(tmp_path):
    steps = ids(run_queue.pending_steps(tmp_path))
    assert steps[0] == "E1 baseline"
    assert steps.index(f"train {E2}") < steps.index(f"eval {E2}") < steps.index("train distance_f1_efficientnet_b0_vessel")
    assert steps.index(f"train {E2}_hA_s2") < steps.index(f"eval {E2}_hA")
    assert steps[-1] == f"noise {E2}"
    assert len(steps) == len(set(steps)) == 20


def test_commands_use_the_runs_folder(tmp_path):
    steps = dict(run_queue.pending_steps(tmp_path, workers=3))
    train = steps["train distance_f1_efficientnet_b0_vessel"]
    assert train[0] == "src/train.py" and train[train.index("--runs-dir") + 1] == str(tmp_path)
    assert train[train.index("--init-from") + 1] == str(tmp_path / E2 / "best.pt")
    assert train[train.index("--workers") + 1] == "3"
    ens = steps[f"eval {E2}_hA"]
    assert ens[1] == str(tmp_path / f"{E2}_hA") and str(tmp_path / f"{E2}_hA_s2") in ens and "--mc" in ens


def test_finished_run_is_skipped_and_started_run_resumes(tmp_path):
    (tmp_path / E2).mkdir()
    (tmp_path / E2 / "done.json").write_text("{}")
    (tmp_path / f"{E2}_hA").mkdir()
    (tmp_path / f"{E2}_hA" / "last.pt").write_bytes(b"")
    steps = dict(run_queue.pending_steps(tmp_path))
    assert f"train {E2}" not in steps and f"eval {E2}" in steps
    assert "--resume" in steps[f"train {E2}_hA"]
    assert "--resume" not in steps[f"train {E2}_hB"]


def test_successful_steps_are_remembered_and_a_failure_stops_the_queue(tmp_path, monkeypatch):
    results = iter([0, 0, 1])
    calls = []
    monkeypatch.setattr(run_queue, "run_step", lambda cmd: calls.append(cmd) or next(results))
    assert run_queue.main(["--runs-dir", str(tmp_path)]) == 1
    assert len(calls) == 3
    left = ids(run_queue.pending_steps(tmp_path))
    assert left[0] == f"eval {E2}" and "E1 baseline" not in left and f"train {E2}" not in left


def test_dry_run_lists_without_running(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_queue, "run_step", lambda cmd: 1 / 0)
    assert run_queue.main(["--runs-dir", str(tmp_path), "--dry-run"]) == 0
    assert f"noise {E2}" in capsys.readouterr().out
