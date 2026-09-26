"""Isolated launch tests: no training processes and no writes to repository logs."""

import asyncio
import builtins
import logging
import os
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import typed_argparse as tap
from fastapi import Request

from ui.backend import server_state
from ui.backend.routes import runners


@pytest.fixture
def experiment_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "experiment"
    directory.mkdir()
    server_state.LightExperiment.json_file(directory).write_text("{}")
    return directory


@pytest.fixture
def launch_state(monkeypatch: pytest.MonkeyPatch) -> server_state.ServerState:
    monkeypatch.setattr(server_state.GarbageCollector, "start", lambda self: None)
    return server_state.ServerState()


@pytest.fixture
def popen(monkeypatch: pytest.MonkeyPatch) -> Mock:
    process = Mock()
    process.wait.side_effect = subprocess.TimeoutExpired(cmd="start_run", timeout=0.5)
    spawn = Mock(return_value=process)
    monkeypatch.setattr(server_state.subprocess, "Popen", spawn)
    return spawn


def test_list_experiments_skips_bad_json_and_keeps_valid_neighbors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setattr(server_state.GarbageCollector, "start", lambda self: None)
    state = server_state.ServerState(str(tmp_path))
    valid = tmp_path / "valid" / "experiment.json"
    valid.parent.mkdir()
    valid.write_text('{"logdir": "valid", "n_steps": 10}')
    broken = tmp_path / "broken" / "experiment.json"
    broken.parent.mkdir()
    broken.write_text("{not json")
    wrong_shape = tmp_path / "wrong_shape" / "experiment.json"
    wrong_shape.parent.mkdir()
    wrong_shape.write_text("[]")
    (tmp_path / "not_an_experiment").mkdir()

    with caplog.at_level(logging.WARNING, logger=server_state.logger.name):
        assert state.list_experiments() == [{"logdir": "valid", "n_steps": 10}]
    assert str(broken) in caplog.text
    assert str(wrong_shape) in caplog.text
    assert "not_an_experiment" not in caplog.text


def test_list_experiments_skips_read_errors_per_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    monkeypatch.setattr(server_state.GarbageCollector, "start", lambda self: None)
    state = server_state.ServerState(str(tmp_path))
    valid = tmp_path / "valid" / "experiment.json"
    valid.parent.mkdir()
    valid.write_text('{"logdir": "valid"}')
    unreadable = tmp_path / "unreadable" / "experiment.json"
    unreadable.parent.mkdir()
    unreadable.write_text('{"logdir": "unreadable"}')
    original_open = builtins.open

    def fail_one_file(path, *args, **kwargs):
        if Path(path) == unreadable:
            raise PermissionError("access denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fail_one_file)
    with caplog.at_level(logging.WARNING, logger=server_state.logger.name):
        assert state.list_experiments() == [{"logdir": "valid"}]
    assert str(unreadable) in caplog.text
    assert "access denied" in caplog.text


def test_default_logs_root_is_not_relative_to_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launch_state: server_state.ServerState):
    monkeypatch.chdir(tmp_path)
    assert launch_state.logdir == str(server_state.PROJECT_ROOT / "logs")
    assert server_state.ServerState(str(tmp_path / "other_logs")).logdir == str(tmp_path / "other_logs")


def test_launch_uses_real_script_and_absolute_experiment_from_any_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launch_state: server_state.ServerState, experiment_dir: Path, popen: Mock
):
    monkeypatch.chdir(tmp_path)
    launch_state.new_runs(str(experiment_dir), 2, 3, 4, 2, disabled_devices=[1], save_actions=False)
    command = popen.call_args.args[0]
    assert command[:3] == [server_state.sys.executable, str(server_state.PROJECT_ROOT / "scripts/start_run.py"), str(experiment_dir)]
    assert "--n-runs=2" in command
    assert "--n-tests=3" in command
    assert "--seed=4" in command
    assert command[-3:] == ["--no-save-actions", "--disabled-devices", "1"]
    assert popen.call_args.kwargs["cwd"] == server_state.PROJECT_ROOT
    assert popen.call_args.kwargs["start_new_session"] is True
    assert popen.call_args.kwargs["stdout"] == subprocess.DEVNULL
    assert popen.call_args.kwargs["stderr"] != subprocess.DEVNULL
    assert popen.return_value.wait.call_args.kwargs["timeout"] <= 3


@pytest.mark.parametrize(
    "changes",
    [
        {"n_runs": 0},
        {"n_runs": True},
        {"n_tests": -1},
        {"n_tests": 1.5},
        {"seed": -1},
        {"seed": "2"},
        {"n_jobs": 0},
        {"test_interval": 0},
        {"gpu_strategy": "invalid"},
        {"save_weights": 1},
        {"save_actions": None},
        {"disabled_devices": "0"},
        {"disabled_devices": [True]},
        {"disabled_devices": [-1]},
        {"disabled_devices": [0, 0]},
        {"device": "cuda:-1"},
        {"device": "mps"},
        {"device": True},
    ],
)
def test_invalid_parameters_never_spawn(changes: dict, launch_state: server_state.ServerState, experiment_dir: Path, popen: Mock):
    params = {"logdir": str(experiment_dir), "n_runs": 1, "n_tests": 1, "seed": 0, "n_jobs": 1}
    params.update(changes)
    with pytest.raises(ValueError):
        launch_state.new_runs(**params)
    popen.assert_not_called()


def test_explicit_gpu_must_exist_and_not_be_disabled(
    monkeypatch: pytest.MonkeyPatch, launch_state: server_state.ServerState, experiment_dir: Path, popen: Mock
):
    monkeypatch.setattr(server_state.torch.cuda, "device_count", lambda: 2)
    for device, disabled in [("cuda:2", []), ("cuda:1", [1]), (0, [0])]:
        with pytest.raises(ValueError, match="disabled or unavailable"):
            launch_state.new_runs(str(experiment_dir), 1, 1, 0, 1, device=device, disabled_devices=disabled)
    popen.assert_not_called()
    launch_state.new_runs(str(experiment_dir), 1, 1, 0, 1, device="cuda:1")
    assert "--device=cuda:1" in popen.call_args.args[0]


def test_relative_experiment_path_resolves_from_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launch_state: server_state.ServerState, experiment_dir: Path, popen: Mock
):
    monkeypatch.chdir(tmp_path)
    launch_state.new_runs(os.path.relpath(experiment_dir, server_state.PROJECT_ROOT), 1, 1, 0, 1)
    assert popen.call_args.args[0][2] == str(experiment_dir)


def test_integer_gpu_index_round_trips_through_script_parser(
    monkeypatch: pytest.MonkeyPatch, launch_state: server_state.ServerState, experiment_dir: Path, popen: Mock
):
    monkeypatch.setattr(server_state.torch.cuda, "device_count", lambda: 2)
    launch_state.new_runs(str(experiment_dir), 2, 3, 4, 2, device=1, disabled_devices=[0], save_actions=False)
    command = popen.call_args.args[0]
    assert "--device=cuda:1" in command
    # run_path does not enter the script's __main__ block or start training.
    arguments = runpy.run_path(str(server_state.START_RUN_SCRIPT))["Arguments"]
    parsed = tap.Parser(arguments).parse_args(command[2:])
    assert parsed.logdir == str(experiment_dir)
    assert parsed.device == "cuda:1"
    assert parsed.disabled_devices == [0]
    assert parsed.n_runs == 2 and parsed.n_jobs == 2
    assert parsed.n_tests == 3 and parsed.seed == 4
    assert parsed.no_save_actions is True


def test_missing_experiment_never_spawns(launch_state: server_state.ServerState, tmp_path: Path, popen: Mock):
    with pytest.raises(FileNotFoundError):
        launch_state.new_runs(str(tmp_path / "missing"), 1, 1, 0, 1)
    popen.assert_not_called()


@pytest.mark.parametrize("returncode, output", [(2, b"bad argument"), (0, b"An error occurred while starting a run: load failed")])
def test_immediate_child_failure_is_reported(
    returncode: int, output: bytes, launch_state: server_state.ServerState, experiment_dir: Path, popen: Mock
):
    def spawn(*args, **kwargs):
        kwargs["stderr"].write(output)
        kwargs["stderr"].flush()
        return SimpleNamespace(wait=lambda timeout: returncode)

    popen.side_effect = spawn
    with pytest.raises(RuntimeError, match="Run launch failed") as exc:
        launch_state.new_runs(str(experiment_dir), 1, 1, 0, 1)
    assert output.decode() in str(exc.value)


def test_spawn_failure_is_not_reported_as_success(launch_state: server_state.ServerState, experiment_dir: Path, popen: Mock):
    popen.side_effect = FileNotFoundError("interpreter unavailable")
    with pytest.raises(RuntimeError, match="Could not start run process"):
        launch_state.new_runs(str(experiment_dir), 1, 1, 0, 1)


def test_still_running_child_is_detached(launch_state: server_state.ServerState, experiment_dir: Path, popen: Mock):
    popen.return_value.wait.side_effect = subprocess.TimeoutExpired(cmd="start_run", timeout=2)
    assert launch_state.new_runs(str(experiment_dir), 1, 1, 0, 1) is None


def test_route_returns_bad_request_before_spawn(monkeypatch: pytest.MonkeyPatch, popen: Mock):
    monkeypatch.setattr(runners, "state", server_state.ServerState.__new__(server_state.ServerState))
    request = SimpleNamespace(json=None)

    async def body():
        return {"nRuns": False, "nTests": 1, "seed": 0}

    request.json = body
    response = asyncio.run(runners.new_run("unused", cast(Request, request)))
    assert response.status_code == 400
    popen.assert_not_called()


def test_route_reports_child_error(monkeypatch: pytest.MonkeyPatch):
    state = Mock()
    state.new_runs.side_effect = RuntimeError("child failed")
    monkeypatch.setattr(runners, "state", state)

    async def body():
        return {"nRuns": 1, "nTests": 1, "seed": 0, "device": "cpu"}

    response = asyncio.run(runners.new_run("experiment", cast(Request, SimpleNamespace(json=body))))
    assert response.status_code == 502
    assert b"child failed" in response.body
    assert state.new_runs.call_args.kwargs["device"] == "cpu"


def test_full_experiment_request_upgrades_cached_lightweight_instance(launch_state, monkeypatch):
    logdir = "/tmp/demo-experiment"
    lightweight = Mock(spec=server_state.LightExperiment)
    full_experiment = Mock(spec=server_state.Experiment)
    calls = []

    def load(path, full=False):
        calls.append((path, full))
        launch_state._experiments[path] = full_experiment if full else lightweight

    monkeypatch.setattr(launch_state, "load_experiment", load)
    assert launch_state.get_experiment(logdir) is lightweight
    assert launch_state.get_experiment(logdir, full=True) is full_experiment
    assert launch_state.get_experiment(logdir, full=True) is full_experiment
    assert calls == [(logdir, False), (logdir, True)]
