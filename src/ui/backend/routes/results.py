from http import HTTPStatus

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse, Response
from . import state
from marl.utils import stats
import orjson

router = APIRouter()


@router.get("/results/load/{logdir:path}")
def get_experiment_results(logdir: str, granularity: int | None = None):
    try:
        exp = state.get_experiment(logdir)
    except (ModuleNotFoundError, FileNotFoundError) as e:
        return PlainTextResponse(str(e), status_code=HTTPStatus.NOT_FOUND)
    metrics = exp.get_experiment_results(granularity=granularity, replace_inf=True)
    # response_data = stats.build_results_payload(metrics, qvalues)
    return Response(orjson.dumps(metrics, option=orjson.OPT_SERIALIZE_NUMPY), media_type="application/json")


@router.get("/results/test/{time_step}/{logdir:path}")
def get_test_results_at(time_step: str, logdir: str):
    exp = state.get_experiment(logdir)
    res = exp.get_tests_at(int(time_step))
    return Response(orjson.dumps(res, option=orjson.OPT_SERIALIZE_NUMPY), media_type="application/json")


@router.get("/results/load-by-run/{logdir:path}")
def get_experiment_results_by_run(logdir: str, granularity: int | None = None):
    runs_results = []
    exp = state.get_experiment(logdir)
    for run in exp.runs:
        active_granularity = granularity if granularity is not None else exp.test_interval
        datasets = stats.compute_datasets([run.test_metrics], logdir, True, category="Test")
        datasets += stats.compute_datasets([run.train_metrics(active_granularity)], logdir, True, category="Train")
        datasets += stats.compute_datasets([run.training_data(active_granularity)], logdir, True, category="Other")
        # qvalues = stats.compute_qvalues([run.qvalues_data(exp.test_interval)], logdir, True, exp.qvalue_labels)
        runs_results.append(datasets)
    return Response(orjson.dumps(runs_results, option=orjson.OPT_SERIALIZE_NUMPY), media_type="application/json")
