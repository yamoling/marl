import orjson
from fastapi import APIRouter
from fastapi.responses import Response


from . import state

router = APIRouter()


@router.get("/results/load/{logdir:path}")
def get_experiment_results(logdir: str, granularity: int | None = None, use_wall_time: bool = False):
    exp = state.get_experiment(logdir)
    aggregate_by = "timestamp_sec" if use_wall_time else "time_step"
    metrics = exp.get_experiment_datasets(granularity=granularity, aggregate_by=aggregate_by)
    return Response(orjson.dumps(metrics, option=orjson.OPT_SERIALIZE_NUMPY), media_type="application/json")


@router.get("/results/test/{time_step}/{logdir:path}")
def get_test_results_at(time_step: str, logdir: str):
    exp = state.get_experiment(logdir)
    res = exp.get_tests_at(int(time_step))
    return Response(orjson.dumps(res, option=orjson.OPT_SERIALIZE_NUMPY), media_type="application/json")
