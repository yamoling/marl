from http import HTTPStatus

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse, Response
from . import state
from marl.utils import stats
import orjson

router = APIRouter()


@router.get("/results/load/{logdir:path}")
def get_experiment_results(logdir: str):
    try:
        exp = state.get_experiment(logdir)
    except (ModuleNotFoundError, FileNotFoundError) as e:
        return PlainTextResponse(str(e), status_code=HTTPStatus.NOT_FOUND)
    metrics = exp.get_experiment_results(replace_inf=True)
    # response_data = stats.build_results_payload(metrics, qvalues)
    return Response(orjson.dumps(metrics), media_type="application/json")


@router.get("/results/test/{time_step}/{logdir:path}")
def get_test_results_at(time_step: str, logdir: str):
    exp = state.get_experiment(logdir)
    res = exp.get_tests_at(int(time_step))
    res = orjson.dumps(res)
    return Response(res, media_type="application/json")


@router.get("/results/load-by-run/{logdir:path}")
def get_experiment_results_by_run(logdir: str):
    runs_results = []
    exp = state.get_experiment(logdir)
    for run in exp.runs:
        datasets = stats.compute_datasets([run.test_metrics], logdir, True, category="Test")
        datasets += stats.compute_datasets([run.train_metrics(exp.test_interval)], logdir, True, category="Train")
        datasets += stats.compute_datasets([run.training_data(exp.test_interval)], logdir, True, category="Other")
        # qvalues = stats.compute_qvalues([run.qvalues_data(exp.test_interval)], logdir, True, exp.qvalue_labels)
        runs_results.append(datasets)
    return Response(orjson.dumps(runs_results), media_type="application/json")
