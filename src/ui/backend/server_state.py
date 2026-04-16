import os
import logging
import subprocess
import sys
import orjson
import time
from threading import Thread
from typing import Optional
from marl.models import Experiment, ReplayEpisode


class ServerState:
    def __init__(self, logdir: str = "logs"):
        self._light_experiments = dict[str, Experiment]()
        self._experiments = dict[str, Experiment]()
        self.last_accessed = dict[str, float]()
        self.logdir = logdir
        GarbageCollector(self).start()

    def list_experiments(self) -> list[dict]:
        experiments = []
        for directory in os.listdir(self.logdir):
            directory = os.path.join(self.logdir, directory)
            try:
                with open(Experiment.json_file(directory)) as f:
                    experiments.append(orjson.loads(f.read()))
            except (FileNotFoundError, NotADirectoryError):
                # Not an experiment directory, ignore
                pass
        return experiments

    def load_experiment(self, logdir: str):
        self._experiments[logdir] = Experiment.load(logdir)

    def new_runs(
        self,
        logdir: str,
        n_runs: int,
        n_tests: int,
        seed: int,
        device: str = "auto",
        gpu_strategy: str = "group",
        disabled_devices: list[int] | None = None,
    ):
        if disabled_devices is None:
            disabled_devices = []
        command = [
            sys.executable,
            "src/start_run.py",
            logdir,
            f"--n-runs={n_runs}",
            f"--n-tests={n_tests}",
            f"--seed={seed}",
            f"--device={device}",
            f"--gpu-strategy={gpu_strategy}",
        ]
        if disabled_devices:
            command.extend(["--disabled-devices", *[str(device_id) for device_id in disabled_devices]])
        logging.info("Starting new process with command: " + " ".join(command))
        # Start a detached training process so runs continue even if the web server exits.
        subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )

    def start_run(self, rundir: str, device: str = "auto"):
        logdir = Experiment.find_experiment_directory(rundir)
        if logdir is None:
            raise FileNotFoundError(f"Could not find experiment for run {rundir}")

        experiment = self.get_experiment(logdir)
        target_run = None
        for run in experiment.runs:
            if run.rundir == rundir:
                target_run = run
                break
        if target_run is None:
            raise FileNotFoundError(f"Could not find run {rundir}")

        if target_run.is_running or target_run.is_completed(experiment.n_steps):
            return
        self.new_runs(logdir, n_runs=1, n_tests=1, seed=target_run.seed, device=device)

    def get_experiment(self, logdir: str) -> Experiment:
        self.last_accessed[logdir] = time.time()
        if logdir not in self._experiments:
            self.load_experiment(logdir)
        return self._experiments[logdir]

    def stop_run(self, rundir: str):
        logdir = Experiment.find_experiment_directory(rundir)
        if logdir is None:
            raise FileNotFoundError(f"Could not find experiment for run {rundir}")
        experiment = self.get_experiment(logdir)
        for run in experiment.runs:
            if run.rundir == rundir:
                run.kill()
                return
        raise FileNotFoundError(f"Could not find run {rundir}")

    def unload_experiment(self, logdir: str) -> Experiment | None:
        return self._experiments.pop(logdir, None)

    def get_runner_port(self, rundir: str) -> Optional[int]:
        return None

    def replay_episode(self, episode_dir: str) -> ReplayEpisode:
        longest_match = ""
        matching_experiment = None
        for logdir, experiment in self._experiments.items():
            if episode_dir.startswith(logdir) and len(logdir) > len(longest_match):
                longest_match = logdir
                matching_experiment = experiment
        if matching_experiment is None:
            raise ValueError(f"Could not find experiment for episode {episode_dir}")
        return matching_experiment.replay_episode(episode_dir)


class GarbageCollector(Thread):
    def __init__(self, state: ServerState, timeout_s: int = 300):
        super().__init__(daemon=True)
        self.state = state
        self.timeout_s = timeout_s

    def run(self):
        while True:
            time.sleep(60)
            to_unload = []
            for logdir, timestamp in self.state.last_accessed.items():
                if time.time() - timestamp > self.timeout_s:
                    to_unload.append(logdir)
            for logdir in to_unload:
                self.state.unload_experiment(logdir)
                del self.state.last_accessed[logdir]
