from flask import Response
from serde.json import to_json
from . import app
from marl.models import Experiment
from marl.models.experiment import ExperimentResults
from marl.utils.stats import round_col


@app.route("/results/load/<path:logdir>", methods=["GET"])
def get_experiment_results(logdir: str):
    results = Experiment.get_experiment_results(logdir, replace_inf=True)
    json_data = to_json(results)
    return Response(json_data, mimetype="application/json")


@app.route("/results/test/<time_step>/<path:logdir>", methods=["GET"])
def get_test_results_at(time_step: str, logdir: str):
    res = Experiment.get_tests_at(logdir, int(time_step))
    res = to_json(res)
    return Response(res, mimetype="application/json")


@app.route("/results/load-by-run/<path:logdir>", methods=["GET"])
def get_experiment_results_by_run(logdir: str):
    runs_results = []
    for run in Experiment.get_runs(logdir):
        test_ticks, test_results = Experiment.compute_datasets([run.test_metrics], True)

        train_ticks, train_results = Experiment.compute_datasets(
            [round_col(run.train_metrics, "time_step", 5000)],
            True,
        )
        # _, train_data = Experiment.compute_datasets([run.training_data], replace_inf=True)
        runs_results.append(ExperimentResults(run.rundir, test_ticks, train_ticks, train_results, test_results))
    return Response(to_json(runs_results), mimetype="application/json")
