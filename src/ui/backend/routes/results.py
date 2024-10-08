from flask import Response
from serde.json import to_json
from . import app, state
from marl.utils import stats


@app.route("/results/load/<path:logdir>", methods=["GET"])
def get_experiment_results(logdir: str):
    try:
        exp = state.get_experiment(logdir)
    except (ModuleNotFoundError, FileNotFoundError) as e:
        return Response(str(e), status=404)
    try:
        results = exp.get_experiment_results(replace_inf=True)
        return Response(to_json(results), mimetype="application/json")
    except Exception as e:
        return Response(str(e), status=500)


@app.route("/results/test/<time_step>/<path:logdir>", methods=["GET"])
def get_test_results_at(time_step: str, logdir: str):
    exp = state.get_experiment(logdir)
    res = exp.get_tests_at(int(time_step))
    res = to_json(res)
    return Response(res, mimetype="application/json")


@app.route("/results/load-by-run/<path:logdir>", methods=["GET"])
def get_experiment_results_by_run(logdir: str):
    runs_results = []
    exp = state.get_experiment(logdir)
    for run in exp.runs:
        datasets = stats.compute_datasets([run.test_metrics], logdir, True, suffix=" [test]")
        datasets += stats.compute_datasets([run.train_metrics(exp.test_interval)], logdir, True, suffix=" [train]")
        datasets += stats.compute_datasets([run.training_data(exp.test_interval)], logdir, True)
        runs_results.append(stats.ExperimentResults(run.rundir, datasets))
    return Response(to_json(runs_results), mimetype="application/json")
