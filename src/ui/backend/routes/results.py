from flask import Response
from serde.json import to_json
from . import app
from marl import Experiment


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
