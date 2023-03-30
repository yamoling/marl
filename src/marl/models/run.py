import os
import json
import shutil
from dataclasses import dataclass
from rlenv.models import Metrics
from marl.utils import CorruptExperimentException
from .replay_episode import ReplayEpisodeSummary
from marl.logging.ws_logger import WSLogger

@dataclass
class Run:
    rundir: str

    def test_metrics(self) -> dict[str, Metrics]:
        metrics: dict[str, list[Metrics]] = {}
        test_dir = os.path.join(self.rundir, "test")
        for time_step in os.listdir(test_dir):
            episode_dir = os.path.join(test_dir, time_step)
            with open(os.path.join(episode_dir, "metrics.json"), "r") as f:
                metrics[time_step] = Metrics(**json.load(f))
        return metrics
    
    @property
    def is_running(self) -> bool:
        return os.path.exists(os.path.join(self.rundir, "pid"))

    @property
    def latest_checkpoint(self) -> str | None:
        # Get the last test directory
        tests = os.listdir(os.path.join(self.rundir, "test"))
        if len(tests) == 0:
            return None
        last_test = max(tests, key=lambda x: int(x))
        return os.path.join(self.rundir, "test", f"{last_test}")

    def delete(self):
        try:
            shutil.rmtree(self.rundir)
            return
        except FileNotFoundError:
            raise CorruptExperimentException(f"Rundir {self.rundir} has already been removed from the file system.")
    
    def get_test_episodes(self, time_step: int) -> list[ReplayEpisodeSummary]:
        episodes = []
        test_dir = os.path.join(self.rundir, "test", f"{time_step}")
        for episode_dir in os.listdir(test_dir):
            episode_dir = os.path.join(test_dir, episode_dir)
            try:
                with open(os.path.join(episode_dir, "metrics.json"), "r") as f:
                    episodes.append(ReplayEpisodeSummary(episode_dir, Metrics(**json.load(f))))
            except (FileNotFoundError, NotADirectoryError):
                # The episode is not yet finished or we have listed a file
                pass
        return episodes
    
    @property
    def current_step(self) -> int:
        try:
            with open(os.path.join(self.rundir, "run.json"), "r") as f:
                return json.load(f)["current_step"]
        except FileNotFoundError:
            return 0


    def stop(self):
        """Stop the run by sending a SIGINT to the process. This method waits for the process to terminate before returning."""
        pid = self.get_pid()
        if pid is not None:
            import signal
            os.kill(pid, signal.SIGINT)
            print("waiting for process to terminate...")
            os.waitpid(pid, 0)
            print(f"process {pid} has terminated")

    def get_port(self) -> int | None:
        try:    
            with open(os.path.join(self.rundir, WSLogger.WS_FILE), "r") as f:
                return int(f.read())
        except FileNotFoundError:
            return None
        
    def get_pid(self) -> int | None:
        try:
            pid_file = os.path.join(self.rundir, "pid")
            with open(pid_file, "r") as f:
                pid = int(f.read())
                os.kill(pid, 0)
                return pid
        except FileNotFoundError:
            return None
        except ProcessLookupError:
            os.remove(pid_file)
        
    def read_status(self) -> dict[str, int]:
        status_path = os.path.join(self.rundir, "run.json")
        with open(status_path, "r") as f:
            return json.load(f)

    def to_json(self) -> dict[str, str | int | None]:
        return {
            "rundir": self.rundir,
            **self.read_status(),
            "port": self.get_port(),
            "pid": self.get_pid()
        }