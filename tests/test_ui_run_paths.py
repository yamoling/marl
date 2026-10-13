"""Regression coverage for serialized run paths when the local UI starts elsewhere."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from marl.models.experiment import LightExperiment
from marl.models.run import LightRun


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    root = tmp_path / "logs" / "demo"
    run_dir = root / "run-7"
    run_dir.mkdir(parents=True)
    run = LightRun(seed=7, rundir="logs/demo/run-7", n_steps=100)
    run.to_file(run_dir / "run.json")
    exp = LightExperiment(n_steps=100, logdir=str(root), loggers=("csv",), creation_timestamp=datetime.now(UTC))
    other_cwd = tmp_path / "src"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    return exp, run_dir


def test_run_is_read_from_discovered_path_not_process_cwd(experiment):
    exp, run_dir = experiment
    run = next(exp.runs)
    assert run.rundir == str(run_dir)
    assert run.run_file == run_dir / "run.json"


@pytest.mark.parametrize("absolute_logdir", [False, True])
def test_nested_relative_run_path_is_accepted(tmp_path, monkeypatch, absolute_logdir):
    """Full relative run paths remain valid for deeply nested experiments. @ai-generated"""
    monkeypatch.chdir(tmp_path)
    root = Path("logs/lle6-layered-500k/dqn/score-0/trial-00")
    run_dir = root / "run-2"
    run_dir.mkdir(parents=True)
    LightRun(seed=2, rundir=str(run_dir), n_steps=500_000).to_file(run_dir / "run.json")
    exp = LightExperiment(
        n_steps=500_000,
        logdir=str(root.resolve() if absolute_logdir else root),
        loggers=("csv",),
        creation_timestamp=datetime.now(UTC),
    )

    runs = list(exp.runs)

    assert len(runs) == 1
    assert runs[0].seed == 2
    assert runs[0].rundir == str(run_dir.resolve())
    assert runs[0].run_file == run_dir.resolve() / "run.json"


def test_unrelated_relative_run_path_is_rejected(experiment):
    """A relative path to another experiment must not pass validation. @ai-generated"""
    exp, run_dir = experiment
    LightRun(seed=7, rundir="logs/other/run-7", n_steps=100).to_file(run_dir / "run.json")
    with pytest.raises(ValueError, match="Run metadata"):
        list(exp.runs)


def test_external_serialized_run_path_is_rejected(experiment, tmp_path):
    exp, run_dir = experiment
    LightRun(seed=7, rundir=str(tmp_path / "outside" / "run-7"), n_steps=100).to_file(run_dir / "run.json")
    with pytest.raises(ValueError, match="Run metadata"):
        list(exp.runs)
