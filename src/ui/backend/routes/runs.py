from . import app, state
from serde.json import to_json
import json



@app.route("/runs/get/<path:logdir>")
def list_runs(logdir: str):
    try:
        exp = state.get_experiment(logdir)
        runs = []
        for run in exp.runs:
            runs.append({
                "rundir": run.rundir,
                "seed": run.seed,
                "progress": run.get_progress(exp.n_steps),
                "pid": run.get_pid(),
            })
        return to_json(runs)
    except (ModuleNotFoundError, AttributeError):
        return json.dumps([])