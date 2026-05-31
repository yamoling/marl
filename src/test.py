import logging
import os
import sys
from datetime import datetime

import dotenv
import pyinstrument
import typed_argparse as tap

from marl import Experiment


class Args(tap.TypedArgs):
    quiet: bool | None = tap.arg("--quiet", default=False)


@pyinstrument.profile()
def to_profile(exp: Experiment):
    runs = []
    for run in exp.runs:
        if run.is_running:
            status = "RUNNING"
        elif run.is_complete:
            status = "COMPLETED"
        elif run.progress == 0:
            status = "CREATED"
        else:
            status = "CANCELLED"
        runs.append(
            {
                "rundir": run.rundir,
                "seed": run.seed,
                "progress": run.progress,
                "pid": run.pid,
                "status": status,
                "n_tests": run.n_tests,
            }
        )
    return runs


def main(args: Args):
    dirs = [os.path.join("logs", logdir) for logdir in os.listdir("logs")]
    start = datetime.now()
    i = 0
    for logdir in dirs:
        exp = Experiment.load(logdir)
        i += len(to_profile(exp))
        break
    end = datetime.now()
    print(i)
    print(f"Time taken: {end - start}")


if __name__ == "__main__":
    dotenv.load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("test.log", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        tap.Parser(Args).bind(main).run()
    except Exception as e:
        logging.error(f"An error occurred while starting a run with command line '{sys.argv}'.\nError: {e}", exc_info=True)
