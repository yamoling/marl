import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from threading import Thread
from typing import Literal, overload

import orjson
import torch

from marl.models import Experiment, LightExperiment, ReplayEpisode

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
START_RUN_SCRIPT = PROJECT_ROOT / "scripts" / "start_run.py"


class ServerState:
    def __init__(self, logdir: str = "logs"):
        """Locate the default logs root independently of the server's cwd. @ai-edited"""
        self._experiments = dict[str, Experiment | LightExperiment]()
        self.last_accessed = dict[str, float]()
        self.logdir = str((PROJECT_ROOT / logdir).resolve())
        GarbageCollector(self).start()

    def list_experiments(self) -> list[dict]:
        """Skip broken experiment files without hiding healthy neighbors. @ai-edited"""
        experiments = []
        for directory in os.listdir(self.logdir):
            json_file = LightExperiment.json_file(os.path.join(self.logdir, directory))
            try:
                with open(json_file, "rb") as f:
                    metadata = orjson.loads(f.read())
                if not isinstance(metadata, dict):
                    logger.warning("Invalid experiment metadata in %s: expected JSON object", json_file)
                    continue
                experiments.append(metadata)
            except (FileNotFoundError, NotADirectoryError):
                # Not an experiment directory, ignore
                pass
            except (orjson.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read experiment metadata from %s: %s", json_file, exc)
        return experiments

    def load_experiment(self, logdir: str, full: bool = False):
        if full:
            self._experiments[logdir] = Experiment.load(logdir)
        else:
            self._experiments[logdir] = LightExperiment.load(logdir)

    def new_runs(
        self,
        logdir: str,
        n_runs: int,
        n_tests: int,
        seed: int,
        n_jobs: int,
        test_interval: int = 5000,
        device: str | int = "auto",
        gpu_strategy: str = "group",
        disabled_devices: list[int] | None = None,
        save_weights: bool = False,
        save_actions: bool = True,
    ):
        """Validate a launch and report child failures during the bounded startup check. @ai-edited"""
        for name, value, minimum in (
            ("n_runs", n_runs, 1),
            ("n_tests", n_tests, 1),
            ("test_interval", test_interval, 1),
            ("n_jobs", n_jobs, 1),
            ("seed", seed, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if gpu_strategy not in ("group", "scatter") or type(gpu_strategy) is not str:
            raise ValueError("gpu_strategy must be 'group' or 'scatter'")
        if type(save_weights) is not bool or type(save_actions) is not bool:
            raise ValueError("save_weights and save_actions must be booleans")
        if disabled_devices is None:
            disabled_devices = []
        if type(disabled_devices) is not list or any(type(index) is not int or index < 0 for index in disabled_devices):
            raise ValueError("disabled_devices must be a list of non-negative GPU indices")
        if len(set(disabled_devices)) != len(disabled_devices):
            raise ValueError("disabled_devices must not contain duplicates")
        if type(device) is int:
            gpu_index = device
        elif type(device) is str and device in ("auto", "cpu", "cuda"):
            gpu_index = 0 if device == "cuda" else None
        elif type(device) is str and re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device):
            gpu_index = int(device[5:])
        else:
            raise ValueError("device must be 'auto', 'cpu', 'cuda', or a non-negative GPU index")
        if gpu_index is not None and (gpu_index < 0 or gpu_index in disabled_devices or gpu_index >= torch.cuda.device_count()):
            raise ValueError(f"GPU {gpu_index} is disabled or unavailable")
        # CLI arguments are strings; --device=0 would not round-trip as an integer.
        device_arg = f"cuda:{device}" if type(device) is int else device

        if type(logdir) is not str or not logdir.strip():
            raise ValueError("logdir must be a non-empty path")
        experiment_dir = (PROJECT_ROOT / logdir).resolve()
        if not LightExperiment.json_file(experiment_dir).is_file():
            raise FileNotFoundError(f"No experiment found at {experiment_dir}")
        command = [
            sys.executable,
            str(START_RUN_SCRIPT),
            str(experiment_dir),
            f"--n-runs={n_runs}",
            f"--n-tests={n_tests}",
            f"--test-interval={test_interval}",
            f"--seed={seed}",
            f"--device={device_arg}",
            f"--gpu-strategy={gpu_strategy}",
            f"--n-jobs={n_jobs}",
        ]
        if save_weights:
            command.append("--save-weights")
        if not save_actions:
            command.append("--no-save-actions")
        if len(disabled_devices) > 0:
            command.extend(["--disabled-devices", *[str(device_id) for device_id in disabled_devices]])
        logger.info("Starting new process with command: %s", " ".join(command))
        # Capture startup errors without buffering the training process's stdout indefinitely.
        with tempfile.TemporaryFile(mode="w+b") as output:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=subprocess.DEVNULL,
                    stderr=output,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    close_fds=True,
                )
            except OSError as exc:
                raise RuntimeError(f"Could not start run process: {exc}") from exc
            try:
                returncode = process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                return  # The child is still starting; it may fail later.
            output.seek(0, os.SEEK_END)
            output.seek(max(0, output.tell() - 8192))
            details = output.read().decode(errors="replace").strip()
            if returncode != 0 or "An error occurred while starting a run" in details:
                logger.error("Run launch failed (exit %s): %s", returncode, details)
                raise RuntimeError(f"Run launch failed (exit {returncode}): {details or 'no output'}")

    def start_run(self, rundir: str, device: str = "auto"):
        logdir = Experiment.find_experiment_directory(Path(rundir))
        if logdir is None:
            raise FileNotFoundError(f"Could not find experiment for run {rundir}")
        logdir = logdir.as_posix()
        experiment = self.get_experiment(logdir)
        target_run = None
        for run in experiment.runs:
            if run.rundir == rundir:
                target_run = run
                break
        if target_run is None:
            raise FileNotFoundError(f"Could not find run {rundir}")

        if target_run.is_running or target_run.is_complete:
            return
        self.new_runs(
            logdir,
            n_runs=1,
            n_tests=1,
            seed=target_run.seed,
            test_interval=target_run.test_interval,
            n_jobs=1,
            device=device,
            save_weights=target_run.save_weights,
            save_actions=target_run.save_actions,
        )

    @overload
    def get_experiment(self, logdir: str | Path, full: Literal[False] = False) -> LightExperiment: ...

    @overload
    def get_experiment(self, logdir: str | Path, full: Literal[True]) -> Experiment: ...

    def get_experiment(self, logdir: str | Path, full: bool = False):
        """Upgrade cached lightweight metadata when an executable specification is required. @ai-edited"""
        if isinstance(logdir, Path):
            logdir = logdir.as_posix()
        self.last_accessed[logdir] = time.time()
        cached = self._experiments.get(logdir)
        if cached is None or (full and not isinstance(cached, Experiment)):
            self.load_experiment(logdir, full=full)
        return self._experiments[logdir]

    def stop_run(self, rundir: str):
        logdir = Experiment.find_experiment_directory(Path(rundir))
        if logdir is None:
            raise FileNotFoundError(f"Could not find experiment for run {rundir}")
        experiment = self.get_experiment(logdir)
        for run in experiment.runs:
            if run.rundir == rundir:
                run.kill()
                return
        raise FileNotFoundError(f"Could not find run {rundir}")

    def unload_experiment(self, logdir: str):
        return self._experiments.pop(logdir, None)

    def replay_episode(self, rundir: str, time_step: int, test_num: int, only_saved_actions: bool) -> ReplayEpisode:
        logdir = Experiment.find_experiment_directory(Path(rundir))
        if logdir is None:
            raise FileNotFoundError(f"Could not find experiment for run {rundir}")
        exp = self.get_experiment(logdir.as_posix(), full=True)
        run = exp.get_run(rundir)
        assert run is not None
        return exp.replay_episode(run.seed, time_step, test_num, only_saved_actions=only_saved_actions)


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
