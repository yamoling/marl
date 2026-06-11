import logging
import os
import sys
from datetime import datetime

import dotenv
import pyinstrument
import typed_argparse as tap
from lle import CooperationLevel

import plot
from marl import Experiment, LightExperiment, Run


class Args(tap.TypedArgs):
    quiet: bool | None = tap.arg("--quiet", default=False)


# @pyinstrument.profile()
def to_profile(exp: LightExperiment):
    runs = []
    for run in exp.runs:
        if isinstance(run, Run):
            raise ValueError("Should not be a run !")
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
    for size in ("100", "250", "500"):
        for coop in CooperationLevel:
            logdirs = [
                os.path.join("logs", d)
                for d in os.listdir("logs")
                if size in d and coop.value in d and os.path.isdir(os.path.join("logs", d))
            ]
            experiments = [LightExperiment.load(logdir) for logdir in logdirs]

            fig, ax = plot.boxplot_at(experiments, "exit_rate")
            ax.set_title(f"Exit rate at size {size} and cooperation level {coop.value}")
            fig.savefig(f"boxplot-{size}-{coop.value}")
            break


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
