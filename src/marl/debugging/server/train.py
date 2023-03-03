from dataclasses import dataclass
import os
import json
import marl
import rlenv
from copy import deepcopy
from laser_env import LaserEnv
from rlenv import Transition
from marl.models import PrioritizedMemory, Experiment
from marl.utils.others import encode_b64_image
from ..replay import replay_episode, replay_video

@dataclass
class MemoryConfig:
    prioritized: bool
    size: int
    nstep: int

@dataclass
class TrainConfig:
    recurrent: bool
    vdn: bool
    env_wrappers: list[str]
    time_limit: int
    level: str
    memory: MemoryConfig

@dataclass
class TrainServerState:
    runner: marl.debugging.DebugRunner | None
    env: rlenv.RLEnv | None
    
    def __init__(self) -> None:
        self.runner = None

    def create_algo(self, config: TrainConfig) -> str:
        """Creates the algorithm and returns its logging directory"""
        if self.runner is not None:
            self.runner.stop = True
        logdir = os.path.join("logs", f"{config.level.replace('/', '_')}")
        if config.memory.prioritized:
            logdir = f"{logdir}-per"
        builder = rlenv.Builder(LaserEnv(config.level))
        for wrapper in config.env_wrappers:
            match wrapper:
                case "TimeLimit": builder.time_limit(config.time_limit)
                case "VideoRecorder": builder.record("videos")
                case "IntrinsicReward": builder.intrinsic_reward("linear", initial_reward=0.5, anneal=10)
                case "AgentId": builder.agent_id()
                case "LogActions": builder.add_logger("action", directory=logdir)
                case other: raise ValueError(f"Unknown wrapper: {wrapper}")
        env, test_env = builder.build_all()
        memory_builder = marl.models.MemoryBuilder(config.memory.size, "episode" if config.recurrent else "transition")
        if config.memory.prioritized:
            memory_builder.prioritized()
        qbuilder = marl.DeepQBuilder(config.recurrent, env)
        if config.vdn:
            qbuilder.vdn()
        qbuilder.memory(memory_builder.build())
        algo = marl.debugging.QLearningDebugger(qbuilder.build(), logdir)
        self.runner = marl.debugging.DebugRunner(env, test_env=test_env, algo=algo, logdir=logdir)
        return logdir

    def get_train_episode(self, episode_num: int) -> dict:
        base_path = os.path.join(self.runner._logger.logdir, "train", f"{episode_num}")
        with open(os.path.join(base_path, "qvalues.json"), "r") as f:
            qvalues = json.load(f)
        with open(os.path.join(base_path, "actions.json"), "r") as f:
            actions = json.load(f)
        episode = replay_episode(deepcopy(self.env), actions)
        json_data = episode.to_json()
        json_data["qvalues"] = qvalues
        return json_data
    
    def get_train_frames(self, episode_num: int) -> dict:
        path = os.path.join(self.runner._logger.logdir, "train", f"{episode_num}", "actions.json")
        with open(path, "r") as f:
            actions = json.load(f)
        frames = replay_video(deepcopy(self.env), actions)
        frames = [encode_b64_image(f) for f in frames]
        return frames

    def get_memory_priorities(self) -> tuple[float, list[float]]:
        # Won't work with DQN
        match self.runner._algo.algo.algo.memory:
            case PrioritizedMemory() as pm:
                p = []
                for i in range(len(pm._tree)):
                    p.append(pm._tree[i])
                return pm._tree.total, p
            case other: return len(other), [1] * len(other)


    def get_transition_from_memory(self, index: int) -> dict:
        # Won't work with DQN
        match self.runner._algo.algo.algo.memory:
            case PrioritizedMemory() as pm:
                transition: rlenv.Transition = pm[index]
                frames = self.replay_episode_from_memory(index, pm, deepcopy(self.env))
                return {
                    **transition.to_json(),
                    "prev_frame": encode_b64_image(frames[0]),
                    "current_frame": encode_b64_image(frames[1])
                }
            case _: return {}

    def _find_start_of_episode(self, index: int, pm: PrioritizedMemory[Transition]) -> int | None:
        while index > 0:
            index -= 1
            if pm[index].done:
                return index + 1
        return None
    
    def replay_episode_from_memory(self, transition_index: int, pm: PrioritizedMemory[Transition], env: rlenv.RLEnv):
        start = self._find_start_of_episode(transition_index, pm)
        if start is None:
            return [None, None]
        env.reset()
        for i in range(start, transition_index):
            actions = pm[i].action
            env.step(actions)
        prev_frame = env.render("rgb_array")
        env.step(pm[transition_index].action)
        frame = env.render("rgb_array")
        return prev_frame, frame

    @property
    def env(self) -> rlenv.RLEnv:
        return self.runner._env
