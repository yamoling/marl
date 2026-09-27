import orjson
from fastapi import APIRouter, Query
from fastapi.responses import Response

from marl.logging import TIME_STEP_COL, TIMESTAMP_COL
from marl.utils import default_serialization

from . import safe_experiment, safe_log_path, safe_run

router = APIRouter()


@router.get("/results/load/{logdir:path}")
def get_experiment_results(logdir: str, granularity: int = Query(gt=0), use_wall_time: bool = False):
    """Load results from a validated experiment directory.

    @ai-edited
    """
    logdir = safe_log_path(logdir)
    exp = safe_experiment(logdir)
    for run in exp.runs:
        safe_run(run, logdir)
    aggregate_by = TIMESTAMP_COL if use_wall_time else TIME_STEP_COL
    metrics = exp.get_results_datasets(granularity=granularity, aggregate_by=aggregate_by)
    return Response(orjson.dumps(metrics, option=orjson.OPT_SERIALIZE_NUMPY), media_type="application/json")


@router.get("/results/test/{time_step}/{logdir:path}")
def get_test_results_at(time_step: str, logdir: str):
    """Load test results from a validated experiment directory.

    @ai-edited
    """
    logdir = safe_log_path(logdir)
    exp = safe_experiment(logdir)
    for run in exp.runs:
        safe_run(run, logdir)
    res = exp.get_tests_at(int(time_step))
    return Response(orjson.dumps(res, option=orjson.OPT_SERIALIZE_NUMPY, default=default_serialization), media_type="application/json")
