import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("no_training_data", [False, True])
def test_collect_runs_restarts_only_missing_or_incomplete_seeds(tmp_path, monkeypatch, dry_run, no_training_data, caplog):
    """Keep completed and active seeds out of the rerun queue. @ai-generated"""
    script = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "run_lle1to5.py"))
    collect_runs = script["collect_runs"]
    monkeypatch.setitem(collect_runs.__globals__, "ALGOS", ("dqn",))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs" / "lle1-dqn").mkdir(parents=True)
    experiment = SimpleNamespace(
        runs=[
            SimpleNamespace(seed=0, is_complete=True, is_running=False),
            SimpleNamespace(seed=1, is_complete=False, is_running=False),
            SimpleNamespace(seed=2, is_complete=False, is_running=True),
        ],
        create_runs=Mock(return_value=["restarted", "new"]),
    )
    monkeypatch.setattr(script["Experiment"], "load", Mock(return_value=experiment))
    args = SimpleNamespace(levels=[1], n_seeds=4, dry_run=dry_run, n_tests=5, test_interval=10_000, no_training_data=no_training_data)
    caplog.set_level("INFO")

    runs = collect_runs(args)

    if dry_run:
        assert runs == []
        experiment.create_runs.assert_not_called()
        assert "would run seeds [1, 3]" in caplog.text
    else:
        assert runs == ["restarted", "new"]
        experiment.create_runs.assert_called_once_with(
            [1, 3], 5, 10_000, save_weights=False, save_actions=False, log_training_data=not no_training_data
        )


def test_collect_runs_skips_completed_experiment(tmp_path, monkeypatch):
    """Do not recreate runs when every requested seed is complete. @ai-generated"""
    script = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "run_lle1to5.py"))
    collect_runs = script["collect_runs"]
    monkeypatch.setitem(collect_runs.__globals__, "ALGOS", ("dqn",))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs" / "lle1-dqn").mkdir(parents=True)
    experiment = SimpleNamespace(
        runs=[SimpleNamespace(seed=0, is_complete=True, is_running=False)],
        create_runs=Mock(),
    )
    monkeypatch.setattr(script["Experiment"], "load", Mock(return_value=experiment))
    args = SimpleNamespace(levels=[1], n_seeds=1, dry_run=False)

    assert collect_runs(args) == []
    experiment.create_runs.assert_not_called()
