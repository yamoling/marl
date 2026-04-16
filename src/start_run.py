from typing import Literal, Optional
import logging
import os
import sys
import dotenv
import typed_argparse as tap


class Arguments(tap.TypedArgs):
    logdir: str = tap.arg(positional=True, help="The experiment directory")
    n_runs: int = tap.arg(default=1, help="Number of runs to create")
    _n_jobs: Optional[int] = tap.arg("--n-jobs", default=None, help="Maximal number of simultaneous processes to use")
    seed: int = tap.arg(default=0, help="The seed for the first run, subsequent ones are incremented by 1")
    n_tests: int = tap.arg(default=1, help="Number of tests to run")
    delay: float = tap.arg(default=5.0, help="Delay in seconds between two consecutive runs")
    _device: Literal["auto", "cpu"] | str = tap.arg("--device", default="auto", help="The device to use (auto, cpu or cuda:<gpu_id>)")
    gpu_strategy: Literal["scatter", "group"] = tap.arg(default="group")
    render: bool = tap.arg(default=False, help="Render the tests")
    disabled_devices: list[int] = tap.arg(default=[], help="Disabled GPU devices", nargs="*")

    @property
    def device(self):
        if self._device in ("auto", "cpu"):
            return self._device
        if self._device.startswith("cuda:"):
            dev_num = int(self._device.split(":")[1])
        else:
            try:
                dev_num = int(self._device)
            except ValueError:
                raise ValueError(f"Invalid device: {self._device}. It should be 'auto', 'cpu' or 'cuda:<gpu_id>'")
        if dev_num in self.disabled_devices:
            raise ValueError(f"Requested device number is {dev_num} but it is present in disabled devices: {self.disabled_devices}")
        return dev_num

    @property
    def n_jobs(self):
        """
        If no value is provided, there are as many processes as there are GPUs.
        If there is no GPU available, then only one process is started.
        """
        if self._n_jobs is not None:
            return min(self._n_jobs, self.n_runs)

        import subprocess

        try:
            # If we have GPUs, then start as many runs as there are GPUs
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

    @property
    def seeds(self):
        return list(range(self.seed, self.seed + self.n_runs))


def main(args: Arguments):
    import marl

    experiment = marl.Experiment.load(args.logdir)
    experiment.run(
        args.seeds,
        fill_strategy=args.gpu_strategy,
        quiet=True,
        device=args.device,
        n_tests=args.n_tests,
        render_tests=args.render,
        n_parallel=args.n_jobs,
    )


if __name__ == "__main__":
    dotenv.load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("start_run.log", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        tap.Parser(Arguments).bind(main).run()
    except Exception as e:
        logging.error(f"An error occurred while starting a run with command line '{sys.argv}'.\nError: {e}", exc_info=True)
