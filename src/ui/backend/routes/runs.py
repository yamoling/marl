from . import app, state
import orjson


@app.route("/runs/get/<path:logdir>")
def list_runs(logdir: str):
    try:
        exp = state.get_experiment(logdir)
        runs = []
        for run in exp.runs:
            runs.append(
                {
                    "rundir": run.rundir,
                    "seed": run.seed,
                    "progress": run.get_progress(exp.n_steps),
                    "pid": run.get_pid(),
                }
            )
        return orjson.dumps(runs)
    except (ModuleNotFoundError, AttributeError):
        return orjson.dumps([])
