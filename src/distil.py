import time
import typed_argparse as tap

from typing import Literal, Optional
from multiprocessing.pool import Pool, AsyncResult


class Arguments(tap.TypedArgs):
    #debug: bool = tap.arg(help="Enable debug mode")
    logdir: str = tap.arg(positional=True, help="The experiment directory")
    n_runs: int = tap.arg(default=1, help="Number of runs to create")
    _n_processes: Optional[int] = tap.arg("--n-processes", default=None, help="Maximal number of simultaneous processes to use")
    seed: int = tap.arg(default=0, help="The seed for the first run, subsequent ones are incremented by 1")
    #delay: float = tap.arg(default=5.0, help="Delay in seconds between two consecutive runs")
    device: Literal["auto", "cpu"] | str = tap.arg(default="auto")
    gpu_strategy: Literal["scatter", "group"] = tap.arg(default="scatter")
    exp_dataset: bool = tap.arg(default="false", help="Will expand the dataset with modified observations changing an agent's position")
    #dataset: Literal["qvalues", "distribution", "action", "all"] = tap.arg(default="distribution", help="Defines what type of output the distilled model should be trained on: qvalues, distribution of actions, action - the single selected action or all - there 3 models with each wil be distilled")
    dataset: Literal["qvalues", "distribution", "action"] = tap.arg(default="distribution", help="Defines what type of output the distilled model should be trained on: qvalues, distribution of actions or action - the single selected action.")

    @property
    def n_processes(self):
        """
        If no value is provided, there are as many processes as there are GPUs.
        If there is no GPU available, then only one process is started.
        """
        if self._n_processes is not None:
            return min(self._n_processes, self.n_runs)

        try:
            # If we have GPUs, then start as many runs as there are GPUs
            import subprocess

            cmd = "nvidia-smi --list-gpus"
            output = subprocess.check_output(cmd, shell=True).decode()
            # The driver exists but no GPU is available (for instance, the eGPU is disconnected)
            if "failed" in output:
                return 1
            n_gpus = int(len(output.splitlines()))
            if n_gpus > 0:
                return min(n_gpus, self.n_runs)
        except subprocess.CalledProcessError:
            pass
        # Otherwise, start only one run at a time on the cpu
        return 1

def start_distillation(args: Arguments):
    import marl

    # Load the experiment from disk and start a child process for each run.
    # The run with seed=0 is spawned in the main process.
    distiler = marl.distilers.DistilHandler.create(args.logdir, args.exp_dataset, args.dataset)
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
