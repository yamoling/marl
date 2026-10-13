import json
from datetime import datetime
from unittest.mock import Mock

import pytest
import torch
from marlenv.catalog import DiscreteMockEnv

from marl import Experiment, algos
from marl.env import EnvConfig
from marl.models import TransitionMemory
from marl.models.run import LightRun, Run
from marl.nn.model_bank import qnetworks
from marl.runners.simple_runner import simple_run


@pytest.fixture
def experiment(tmp_path):
    """Build a tiny experiment isolated from repository logs. @ai-generated"""
    env = EnvConfig.from_any(DiscreteMockEnv())
    return Experiment(
        n_steps=3,
        logdir=tmp_path.as_posix(),
        loggers=("csv",),
        creation_timestamp=datetime.now(),
        trainer=algos.DQN(qnetworks.from_env(env), memory=TransitionMemory(10), mixer=None),
        env=env,
        test_env=env,
    )


@pytest.mark.parametrize("enabled", [True, False])
def test_training_data_logging_flag_is_persisted(experiment, enabled):
    """Both worker and lightweight loads preserve the run recording choice. @ai-generated"""
    run = experiment.create_runs([0], 1, 2, False, False, log_training_data=enabled)[0]
    assert run.log_training_data is enabled
    assert Run.load(run.runpath).log_training_data is enabled
    assert LightRun.load(run.runpath).log_training_data is enabled


def test_legacy_run_defaults_to_logging_training_data(experiment):
    """Old run files without the new flag retain their original behaviour. @ai-generated"""
    run = experiment.create_runs([0], 1, 2, False, False)[0]
    assert run.log_training_data is True
    data = json.loads(run.run_file.read_text())
    del data["log_training_data"]
    run.run_file.write_text(json.dumps(data))
    assert Run.load(run.runpath).log_training_data is True
    assert LightRun.load(run.runpath).log_training_data is True


@pytest.mark.parametrize("enabled", [True, False])
def test_runner_training_data_logging_does_not_skip_updates(experiment, enabled):
    """Suppress only update metrics, retaining learning and episode/test CSVs. @ai-generated"""
    run = experiment.create_runs([0], 1, 2, False, False, log_training_data=enabled)[0]
    run.trainer.compile = Mock()
    run.trainer.update_step = Mock(return_value={"step_loss": 1.0})
    run.trainer.update_episode = Mock(return_value={"episode_loss": 2.0})

    simple_run(run, quiet=True, render_tests=False, device=torch.device("cpu"))

    assert run.trainer.update_step.call_count == run.n_steps
    assert run.trainer.update_episode.call_count >= 1
    assert (run.runpath / "train.csv").exists()
    assert (run.runpath / "test.csv").exists()
    training_data = run.runpath / "training_data.csv"
    assert training_data.exists() is enabled
    if enabled:
        content = training_data.read_text()
        assert "step_loss" in content
        assert "episode_loss" in content


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_experiment_run_forwards_training_data_logging(experiment, monkeypatch, enabled, n_jobs):
    """The convenience API forwards the flag for sequential and parallel runs. @ai-generated"""
    runner = Mock()
    monkeypatch.setattr("marl.runners.sequential_run", runner)
    monkeypatch.setattr("marl.runners.parallel_run", runner)

    experiment.run(seeds=2, device="cpu", n_jobs=n_jobs, log_training_data=enabled)

    runner.assert_called_once()
    runs = runner.call_args.args[0]
    assert len(runs) == 2
    assert all(run.log_training_data is enabled for run in runs)
