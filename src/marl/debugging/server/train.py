from dataclasses import dataclass
import os
import json
import marl
import rlenv
from copy import deepcopy
from laser_env import LaserEnv
from ..replay import replay_episode, replay_video

ALGORITHMS = [ "DQN", "VDN linear"]
ENV_WRAPPERS = ["TimeLimit", "VideoRecorder", "IntrinsicReward", "AgentId"]

@dataclass
class TrainServerState:
    runner: marl.debugging.DebugRunner
    
    def __init__(self) -> None:
        self.runner = None

    def create_algo(
        self,
        algo_name: str, 
        map_file: str, 
        wrappers: list[str], 
        prioritized: bool, 
        memory_size: int, 
        time_limit: int|None
    ):
        if self.runner is not None:
            self.runner.stop = True
        logdir = os.path.join("logs", f"{map_file.replace('/', '_')}-{algo_name}")
        if prioritized:
            logdir = f"{logdir}-per"
        builder = rlenv.Builder(LaserEnv(map_file))
        for wrapper in wrappers:
            match wrapper:
                case "TimeLimit": builder.time_limit(time_limit)
                case "VideoRecorder": builder.record("videos")
                case "IntrinsicReward": builder.intrinsic_reward("linear", initial_reward=0.5, anneal=10)
                case "AgentId": builder.agent_id()
                case other: raise ValueError(f"Unknown wrapper: {wrapper}")
        builder.add_logger("action", directory=logdir)
        env, test_env = builder.build_all()
        memory = marl.models.TransitionMemory(memory_size)
        if prioritized:
            memory = marl.models.PrioritizedMemory(memory, alpha=0.6, beta=0.4)
        qnetwork = marl.nn.model_bank.MLP.from_env(env)
        algo = marl.qlearning.DQN(qnetwork=qnetwork, memory=memory)
        if algo_name == "VDN linear":
            algo = marl.qlearning.vdn.VDN(algo)
        algo = marl.debugging.QLearningDebugger(algo, logdir)
        self.runner = marl.debugging.DebugRunner(env, test_env=test_env, algo=algo, logdir=logdir)

    def get_train_episode(self, episode_num: int) -> dict:
        base_path = os.path.join(self.runner._logger.logdir, "train", f"{episode_num}")
        with open(os.path.join(base_path, "qvalues.json"), "r") as f:
            qvalues = json.load(f)
        with open(os.path.join(base_path, "actions.json"), "r") as f:
            actions = json.load(f)
        episode = replay_episode(deepcopy(self.runner._env), actions)
        json_data = episode.to_json()
        json_data["qvalues"] = qvalues
        return json_data
    
    def get_train_frames(self, episode_num: int) -> dict:
        path = os.path.join(self.runner._logger.logdir, "train", f"{episode_num}", "actions.json")
        with open(path, "r") as f:
            actions = json.load(f)
        frames = replay_video(deepcopy(self.runner._env), actions)
        frames = [self.runner.encode_frame(f) for f in frames]
        return frames
