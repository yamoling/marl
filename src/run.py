import time
from lle import LLE
from marlenv import MultiDiscreteSpace, RLEnvWrapper
import marlenv
import typed_argparse as tap

from typing import Literal, Optional
from multiprocessing.pool import Pool, AsyncResult

import numpy as np
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class ShapedLabyrinth(RLEnvWrapper[MultiDiscreteSpace]):
    def __init__(self, delay: int):
        self.delay = delay
        self._key_pos = [(7, 1), (0, 3), (7, 5), (0, 7), (7, 9)]
        self._reward_countdown = OrderedDict.fromkeys(self._key_pos, delay)
        lle = LLE.from_file("maps/tmp").obs_type("layered").state_type("state").build()
        self._world = lle.world
        env = marlenv.Builder(lle).time_limit(90 + delay).build()
        super().__init__(
            env,
            extra_shape=(5 + env.extras_shape[0],),
            extra_meanings=env.extras_meanings + ["checkpoint 0", "checkpoint 1", "checkpoint 2", "checkpoint 3", "checkpoint 4"],
        )

    def reset(self):
        self._reward_countdown = OrderedDict.fromkeys(self._key_pos, self.delay)
        _, state = super().reset()
        return self.get_observation(), state

    def get_observation(self):
        obs = super().get_observation()
        extra = np.array([[r for r in self._reward_countdown.values()]], dtype=np.float32)
        if self.delay != 0:
            extra = (extra / self.delay).astype(np.float32)
        obs.add_extra(extra)
        return obs

    def step(self, actions):
        step = super().step(actions)
        agent_pos = self._world.agents_positions[0]
        if agent_pos in self._reward_countdown:
            countdown = self._reward_countdown[agent_pos]
            if countdown == 0:
                step.reward += 1.0
            self._reward_countdown[agent_pos] = max(-1, countdown - 1)
        step.obs = self.get_observation()
        if step.done:
            # Flush rewards for the final step
            for countdown in self._reward_countdown.values():
                if countdown >= 0 and countdown != self.delay:
                    step.reward += 1.0
        return step


def make_lle(delay: int):
    # env = LLE.from_file("maps/tmp").obs_type("layered").state_type("state").build()
    if delay >= 0:
        env = ShapedLabyrinth(delay)
    else:
        env = LLE.from_file("maps/tmp").obs_type("layered").state_type("state").builder().time_limit(90).build()
    # env = marlenv.Builder(env).time_limit(90).build()
    test_env = None
    return env, test_env


class Arguments(tap.TypedArgs):
    debug: bool = tap.arg(help="Enable debug mode")
    logdir: str = tap.arg(positional=True, help="The experiment directory")
    n_runs: int = tap.arg(default=1, help="Number of runs to create")
    _n_processes: Optional[int] = tap.arg("--n-processes", default=None, help="Maximal number of simultaneous processes to use")
    seed: int = tap.arg(default=0, help="The seed for the first run, subsequent ones are incremented by 1")
    n_tests: int = tap.arg(default=1, help="Number of tests to run")
    delay: float = tap.arg(default=5.0, help="Delay in seconds between two consecutive runs")
    _device: Literal["auto", "cpu"] | str = tap.arg("--device", default="auto", help="The device to use (auto, cpu or cuda:<gpu_id>)")
    gpu_strategy: Literal["scatter", "group"] = tap.arg(default="scatter")
    render: bool = tap.arg(default=False, help="Render the tests")

    @property
    def device(self) -> Literal["auto", "cpu"] | int:
        if self._device in ("auto", "cpu"):
            return self._device
        try:
            return int(self._device)
        except ValueError:
            pass
        if self._device.startswith("cuda:"):
            return int(self._device.split(":")[1])
        raise ValueError(f"Invalid device: {self._device}. It should be 'auto', 'cpu' or 'cuda:<gpu_id>'")

    @property
    def n_processes(self):
        """
        If no value is provided, there are as many processes as there are GPUs.
        If there is no GPU available, then only one process is started.
        """
        if self._n_processes is not None:
            return min(self._n_processes, self.n_runs)

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


def start_run(args: Arguments, run_num: int, estimated_gpu_memory: int):
    import marl

    # Load the experiment from disk and start a child process for each run.
    # The run with seed=0 is spawned in the main process.
    experiment = marl.Experiment.load(args.logdir)
    experiment.run(
        seed=args.seed + run_num,
        fill_strategy=args.gpu_strategy,
        required_memory_MB=estimated_gpu_memory,
        quiet=run_num > 0,
        device=args.device,
        n_tests=args.n_tests,
        render_tests=args.render,
    )


def main(args: Arguments):
    if args.debug or args.n_runs == 1:
        start_run(args, 0, 0)
        return

    from marl.utils.gpu import get_max_gpu_usage, get_gpu_processes

    # NOTE: within a docker, the pids do not match with the host, so we have to retrieve the pids "unreliably"
    initial_pids = get_gpu_processes()
    estimated_gpu_memory = 0
    print(f"Running on {args.n_processes} processes")
    with Pool(args.n_processes) as pool:
        handles = list[AsyncResult]()
        for run_num in range(args.n_runs):
            h = pool.apply_async(start_run, (args, run_num, estimated_gpu_memory))
            handles.append(h)
            # If it is not the last process, wait a bit to let the time to allocate the GPUs correctly.
            if run_num != args.n_runs - 1:
                time.sleep(args.delay)
                new_pids = get_gpu_processes() - initial_pids
                estimated_gpu_memory = get_max_gpu_usage(new_pids)

        for h in handles:
            h.get()


if __name__ == "__main__":
    tap.Parser(Arguments).bind(main).run()
