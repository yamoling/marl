import time
import typed_argparse as tap

from typing import Literal, Optional
from multiprocessing.pool import Pool, AsyncResult

from marl.xmarl.distilers import DistilHandler  


class Arguments(tap.TypedArgs):
    #debug: bool = tap.arg(help="Enable debug mode")
    logdir: str = tap.arg(positional=True, help="The experiment directory")
    n_runs: int = tap.arg(default=1, help="Number of runs to consider")
    n_sets: int = tap.arg(default=5, help="Number of tests to consider for each run")
    epochs: int = tap.arg(default=10, help="Number of epochs to run")
    seed: int = tap.arg(default=0, help="The seed for the first run, subsequent ones are incremented by 1")
    #delay: float = tap.arg(default=5.0, help="Delay in seconds between two consecutive runs")
    #device: Literal["auto", "cpu"] | str = tap.arg(default="auto")
    gpu_strategy: Literal["scatter", "group"] = tap.arg(default="scatter")
    #exp_dataset: bool = tap.arg(default=False, help="Will expand the dataset with modified observations changing an agent's position")
    extras: bool = tap.arg(default=False, help="Will add the extras from env to the input AND! the agent's position on the grid (get when flattening)")
    #output: Literal["qvalues", "distribution", "action"] = tap.arg(default="distribution", help="Defines what type of output the distilled model should be trained on: qvalues, distribution of actions or action - the single selected action.") # Rather should be defined by the model?
    input: Literal["full_obs", "abstracted_obs"] = tap.arg(default="full_obs", help="Defines what type of input the distilled model should be trained on: full-obs is the standard full observation (automatically flattened to be 2D), abstracted-obs is an abstracted version of the observation containing heuristical elements of the observation (i.e.: distance from agent-0 to agent-1).") # WOuld be a good option to put "partial" view here, for the work's sake I made a dataset from an experiment which ran in partial
    distiller: Literal["sdt"] = tap.arg(default="sdt", help="Defines what type of model we want to distil to.")
    # If time or future work: Use same structure as experiments, with distil being equivalent to run and a create_distillation file to create the dataset, model etc... allowing more parametrisation of distillers    
    individual: bool = tap.arg(default=False, help="Defines whether the distilled model should consider all agents, True: individually (1 distilled model per agent). False: as one (one distilled model, agent povs as batches).")
    importance_perc: int = tap.arg(default=95, help="The value of the percentile to use when filtering the data by importance. A value of 95, keeps the 95th percentile, so the top 5% most important data.")

def start_distillation(args: Arguments):
    # Load the experiment from disk and start a child process for each run.
    # The run with seed=0 is spawned in the main process.
    if not args.individual and args.input == "abstracted_obs": raise Exception("Can't use abstracted obs in compounded view, abstracted view by default agent based!")
    distiler = DistilHandler.create(args.logdir, args.n_runs, args.n_sets, args.epochs, args.distiller, args.input, args.extras, args.individual, args.importance_perc)
    distiler.run()


def main(args: Arguments):
    #if args.debug:
    #    start_distillation(args, 0, 0)
    #    return

    #from marl.utils.gpu import get_max_gpu_usage, get_gpu_processes

    # NOTE: within a docker, the pids do not match with the host, so we have to retrieve the pids "unreliably"
    #initial_pids = get_gpu_processes()
    #estimated_gpu_memory = 0
    start_distillation(args)


if __name__ == "__main__":
    tap.Parser(Arguments).bind(main).run()
