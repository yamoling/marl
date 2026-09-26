"""
Launching and restarting runs, ported from tests/test_ui_run_launch.py.
No training process is started: the launcher's Popen is mocked.
"""

import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest
import typed_argparse as tap

from studio.backend import settings
from studio.backend.services import launcher

BODY = {"n_runs": 2, "seed": 5, "n_tests": 3, "test_interval": 1000, "n_jobs": 2, "save_actions": False, "disabled_devices": [1]}


def test_launch_defaults(api):
    healthy = api.get("/api/experiments/healthy/launch-defaults").json()
    assert healthy["next_seed"] == 3 and healthy["existing_seeds"] == [0, 1, 2]
    assert (healthy["n_tests"], healthy["test_interval"], healthy["save_weights"], healthy["save_actions"]) == (2, 1000, False, True)
    assert healthy["capabilities"]["launch"] is True
    broken = api.get("/api/experiments/unknown-class/launch-defaults").json()
    assert broken["capabilities"]["launch"] is False
    assert "deserialize-failed" in {i["code"] for i in broken["issues"]}
    light = api.get("/api/experiments/light/launch-defaults").json()
    assert light["capabilities"]["launch"] is False and light["next_seed"] == 1


@pytest.mark.parametrize(("scenario", "code"), [("unknown-class", "deserialize-failed"), ("light", "missing-keys")])
def test_not_launchable_is_409(api, popen, scenario, code):
    response = api.post(f"/api/experiments/{scenario}/runs", json={"n_runs": 1, "seed": 10, "n_tests": 1})
    assert response.status_code == 409
    assert response.json()["error"] == "not-launchable" and response.json()["issue"]["code"] == code
    popen.assert_not_called()


def test_seed_collision_is_409(api, popen):
    response = api.post("/api/experiments/healthy/runs", json={"n_runs": 3, "seed": 1, "n_tests": 1})
    assert response.status_code == 409
    assert response.json()["error"] == "seed-collision" and "1, 2" in response.json()["message"]
    popen.assert_not_called()


def test_collision_is_rechecked_right_before_spawn(mut_api, popen, monkeypatch):
    http, root = mut_api
    original = launcher.command

    def racing_command(*args):
        (root / "healthy" / "run-5").mkdir()  # Another launch created the run meanwhile
        return original(*args)

    monkeypatch.setattr(launcher, "command", racing_command)
    assert http.post("/api/experiments/healthy/runs", json=BODY).status_code == 409
    popen.assert_not_called()


def test_launch_spawns_the_real_script(mut_api, popen, tmp_path, monkeypatch):
    http, root = mut_api
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher, "cuda_device_count", lambda: 2)
    response = http.post("/api/experiments/healthy/runs", json=BODY | {"device": "cpu"})
    assert response.status_code == 202, response.text
    assert response.json() == {"runs": ["healthy/run-5", "healthy/run-6"]}
    command = popen.call_args.args[0]
    assert command[:3] == [sys.executable, str(settings.START_RUN_SCRIPT), str(root / "healthy")]
    for arg in ("--n-runs=2", "--n-tests=3", "--seed=5", "--test-interval=1000", "--n-jobs=2", "--device=cpu", "--gpu-strategy=group"):
        assert arg in command
    assert command[-3:] == ["--no-save-actions", "--disabled-devices", "1"]
    kwargs = popen.call_args.kwargs
    assert kwargs["cwd"] == settings.PROJECT_ROOT and kwargs["start_new_session"] is True
    assert kwargs["stdout"] == subprocess.DEVNULL and kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["stderr"] not in (subprocess.DEVNULL, None)
    assert popen.return_value.wait.call_args_list[0].kwargs["timeout"] <= 3


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
        {"unknown": 1},
    ],
)
def test_invalid_parameters_never_spawn(api, popen, changes):
    response = api.post("/api/experiments/healthy/runs", json={"n_runs": 1, "n_tests": 1, "seed": 100, "n_jobs": 1} | changes)
    assert response.status_code == 400, response.text
    assert response.json()["error"] == "invalid-request"
    popen.assert_not_called()


@pytest.mark.parametrize("body", [None, [], {"n_runs": 1, "seed": 100}])
def test_malformed_bodies_are_400(api, popen, body):
    assert api.post("/api/experiments/healthy/runs", json=body).status_code == 400
    popen.assert_not_called()


def test_explicit_gpu_must_exist_and_not_be_disabled(api, popen, monkeypatch):
    monkeypatch.setattr(launcher, "cuda_device_count", lambda: 2)
    for device, disabled in [("cuda:2", []), ("cuda:1", [1]), ("cuda", [0])]:
        response = api.post(
            "/api/experiments/healthy/runs", json={"n_runs": 1, "n_tests": 1, "seed": 100, "device": device, "disabled_devices": disabled}
        )
        assert response.status_code == 400 and "disabled or unavailable" in response.json()["message"]
    popen.assert_not_called()


def test_integer_gpu_index_round_trips_through_script_parser(mut_api, popen, monkeypatch):
    http, root = mut_api
    monkeypatch.setattr(launcher, "cuda_device_count", lambda: 2)
    assert http.post("/api/experiments/healthy/runs", json=BODY | {"device": 1, "disabled_devices": [0]}).status_code == 202
    command = popen.call_args.args[0]
    assert "--device=cuda:1" in command
    # run_path does not enter the script's __main__ block or start training.
    arguments = runpy.run_path(str(settings.START_RUN_SCRIPT))["Arguments"]
    parsed = tap.Parser(arguments).parse_args(command[2:])
    assert parsed.logdir == str(root / "healthy") and parsed.device == "cuda:1" and parsed.disabled_devices == [0]
    assert (parsed.n_runs, parsed.n_jobs, parsed.n_tests, parsed.seed, parsed.no_save_actions) == (2, 2, 3, 5, True)


@pytest.mark.parametrize(("returncode", "output"), [(2, b"bad argument"), (0, b"An error occurred while starting a run: load failed")])
def test_immediate_child_failure_is_502_with_stderr(api, popen, returncode, output):
    def spawn(*args, **kwargs):
        kwargs["stderr"].write(output)
        kwargs["stderr"].flush()
        return SimpleNamespace(wait=lambda timeout: returncode)

    popen.side_effect = spawn
    response = api.post("/api/experiments/healthy/runs", json={"n_runs": 1, "n_tests": 1, "seed": 100})
    assert response.status_code == 502
    assert response.json()["error"] == "launch-failed" and output.decode() in response.json()["detail"]


def test_spawn_failure_is_502(api, popen):
    popen.side_effect = FileNotFoundError("interpreter unavailable")
    response = api.post("/api/experiments/healthy/runs", json={"n_runs": 1, "n_tests": 1, "seed": 100})
    assert response.status_code == 502 and "interpreter unavailable" in response.json()["detail"]


def test_restart_rules(mut_api, popen):
    http, root = mut_api
    # A completed run cannot be restarted
    response = http.post("/api/runs/healthy/run-0/restart", json={})
    assert response.status_code == 409 and response.json()["error"] == "not-restartable"
    # A created run (no table rows) can, with its own configuration
    for name in ("test.csv", "train.csv", "training_data.csv"):
        (root / "healthy" / "run-2" / name).unlink()
    assert http.post("/api/runs/healthy/run-2/restart").status_code == 202
    command = popen.call_args.args[0]
    assert "--seed=2" in command and "--n-tests=2" in command and "--test-interval=1000" in command
    # Not launchable experiments cannot restart runs
    for name in ("test.csv", "train.csv", "training_data.csv"):
        (root / "unknown-class" / "run-0" / name).unlink()
    assert http.post("/api/runs/unknown-class/run-0/restart").status_code == 409
    assert http.post("/api/runs/healthy/run-2/restart", json={"gpu": 1}).status_code == 400
    assert popen.call_count == 1
