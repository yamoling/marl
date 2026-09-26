"""Regression coverage for serialized run paths when the local UI starts elsewhere."""

from datetime import UTC, datetime

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


def test_external_serialized_run_path_is_rejected(experiment, tmp_path):
    exp, run_dir = experiment
    LightRun(seed=7, rundir=str(tmp_path / "outside" / "run-7"), n_steps=100).to_file(run_dir / "run.json")
    with pytest.raises(ValueError, match="Run metadata"):
        list(exp.runs)
