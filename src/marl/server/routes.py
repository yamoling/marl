from http import HTTPStatus
from flask import request
from .messages import (
    MemoryConfig,
    ExperimentConfig,
    GeneratorConfig,
    RunConfig,
    TrainConfig
)
from marl.server import app, state
from marl.utils import CorruptExperimentException
from marl.utils import exceptions


@app.route("/experiment/replay/<path:path>")
def get_episode(path: str):
    return state.replay_episode(path).to_json()


@app.route("/experiment/test/list/<time_step>/<path:directory>", methods=["GET"])
def list_test_episodes(time_step: str, directory: str):
    return [e.to_json() for e in state.get_test_episodes_at(directory, int(time_step))]


@app.route("/runner/create/<path:logdir>", methods=["POST"])
def create_runner(logdir: str):
    run_config = RunConfig(**request.get_json())
    state.create_runner(logdir, run_config)
    return ""


@app.route("/runner/port/<path:rundir>", methods=["GET"])
def get_runner_port(rundir: str):
    port = state.get_runner_port(rundir)
    if port is None:
        return "", HTTPStatus.NOT_FOUND
    return str(port)


@app.route("/runner/restart/<path:rundir>", methods=["POST"])
def restart_runner(rundir: str):
    data = TrainConfig(**request.get_json())
    state.restart_runner(rundir, data)
    return ""


@app.route("/runner/stop/<path:rundir>", methods=["POST"])
def stop_runner(rundir: str):
    state.stop_runner(rundir)
    return ""


@app.route("/runner/delete/<path:rundir>", methods=["DELETE"])
def delete_runner(rundir: str):
    state.delete_runner(rundir)
    return ""


@app.route("/experiment/create", methods=["POST"])
def create_experiment():
    data: dict = request.get_json()
    data["level"] = data["level"]
    data["memory"] = MemoryConfig(**data["memory"])
    data["generator"] = GeneratorConfig(**data["generator"])
    if data["forced_actions"] is not None:
        data["forced_actions"] = {
            int(key): value for key, value in data["forced_actions"].items()
        }
    train_config = ExperimentConfig(**data)
    try:
        exp = state.create_experiment(train_config)
        return exp.summary()
    except Exception as e:
        return str(e), HTTPStatus.INTERNAL_SERVER_ERROR


@app.route("/experiment/list")
def list_experiments():
    try:
        return [e.summary() for e in state.list_experiments().values()]
    except exceptions.ExperimentVersionMismatch as e:
        print(e)
        return str(e), HTTPStatus.INTERNAL_SERVER_ERROR


@app.route("/experiment/load/<path:logdir>", methods=["GET"])
def load_experiment(logdir: str):
    try:
        experiment = state.load_experiment(logdir)
        res = experiment.summary()
        time_steps, datasets = experiment.test_metrics()
        res["test_metrics"] = {
            "time_steps": time_steps,
            "datasets": [d.to_json() for d in datasets],
        }
        time_steps, datasets = experiment.train_metrics()
        res["train_metrics"] = {
            "time_steps": time_steps,
            "datasets": [d.to_json() for d in datasets],
        }
        return res
    except CorruptExperimentException as e:
        print(e)
        return str(e), HTTPStatus.INTERNAL_SERVER_ERROR


@app.route("/experiment/load/<path:logdir>", methods=["DELETE"])
def unload_experiment(logdir: str):
    state.unload_experiment(logdir)
    return ""


@app.route("/experiment/delete/<path:logdir>", methods=["DELETE"])
def delete_experiment(logdir: str):
    try:
        state.delete_experiment(logdir)
        return ""
    except ValueError as e:
        return str(e), HTTPStatus.BAD_REQUEST
