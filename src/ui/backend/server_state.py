import os
import subprocess
import time
from threading import Thread
from typing import Optional
from marl.models import Experiment, ReplayEpisode


class ServerState:
    def __init__(self, logdir="logs"):
        self._experiments = dict[str, Experiment]()
        self.last_accessed = dict[str, float]()
        self.logdir = logdir
        GarbageCollector(self).start()

    def list_experiments(self) -> list[dict]:
        experiments = []
        for directory in os.listdir(self.logdir):
            directory = os.path.join(self.logdir, directory)
            try:
                experiments.append(Experiment.get_parameters(directory))
            except FileNotFoundError:
                # Not an experiment directory, ignore
                pass
        return experiments

    def load_experiment(self, logdir: str):
        self._experiments[logdir] = Experiment.load(logdir)

    def new_runs(self, logdir: str, n_runs: int, n_tests: int, seed: int):
        command = f"python src/run.py {logdir} --n-runs={n_runs} --n-tests={n_tests} --seed={seed} --device=auto"
        print(command)
        return
        # Start a completely detached new run
        p = subprocess.Popen(command.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def get_experiment(self, logdir: str) -> Experiment:
        self.last_accessed[logdir] = time.time()
        if logdir not in self._experiments:
            self.load_experiment(logdir)
        return self._experiments[logdir]

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
