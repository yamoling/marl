from http import HTTPStatus
from flask import Response
from . import app, state
from marl.utils import stats
import orjson


@app.route("/results/load/<path:logdir>", methods=["GET"])
def get_experiment_results(logdir: str):
    try:
        exp = state.get_light_experiment(logdir)
    except (ModuleNotFoundError, FileNotFoundError) as e:
        return Response(str(e), status=HTTPStatus.NOT_FOUND)
    try:
        results = exp.get_experiment_results(replace_inf=True)
        return Response(orjson.dumps(results), mimetype="application/json")
    except Exception as e:
        print(e)
        return Response(str(e), status=500)


@app.route("/results/test/<time_step>/<path:logdir>", methods=["GET"])
def get_test_results_at(time_step: str, logdir: str):
    exp = state.get_light_experiment(logdir)
    res = exp.get_tests_at(int(time_step))
    res = orjson.dumps(res)
    return Response(res, mimetype="application/json")


@app.route("/results/load-by-run/<path:logdir>", methods=["GET"])
def get_experiment_results_by_run(logdir: str):
    runs_results = []
    exp = state.get_light_experiment(logdir)
    for run in exp.runs:
        datasets = stats.compute_datasets([run.test_metrics], logdir, True, suffix=" [test]")
        datasets += stats.compute_datasets([run.train_metrics], logdir, True, suffix=" [train]")
        datasets += stats.compute_datasets([run.training_data], logdir, True)
        runs_results.append(stats.ExperimentResults(run.rundir, datasets))
    return Response(orjson.dumps(runs_results), mimetype="application/json")
