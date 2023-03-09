from dataclasses import dataclass
import os
import json
import marl
import rlenv
from copy import deepcopy
from laser_env import LaserEnv
import laser_env
from rlenv import Transition
from marl.models import PrioritizedMemory
from marl.utils.others import encode_b64_image, get_available_port

from ..replay import replay_episode, replay_video
from .messages import TrainConfig, StartTrain


@dataclass
class TrainServerState:
    runner: marl.Runner | None
    env: rlenv.RLEnv | None
    
    def __init__(self) -> None:
        self.runner = None

    def create_runner(self, config: TrainConfig):
        """Creates the algorithm and returns its logging directory"""
        logger = marl.logging.WSLogger(config.logdir)
        env, test_env = self._create_env(config)
        memory = self._create_memory(config)
        qbuilder = marl.DeepQBuilder()
        if config.vdn:
            qbuilder.vdn()
        qbuilder.qnetwork_default(env)
        qbuilder.memory(memory)
        algo = marl.wrappers.ReplayWrapper(qbuilder.build(), logger.logdir)
        self.runner = marl.Runner(env, test_env=test_env, algo=algo, logger=logger)
        return logger.port
    
    @staticmethod
    def _create_env(config: TrainConfig):
        if config.static_map:
            env = LaserEnv(config.level)
        else:
            generator = (laser_env.LevelGenerator(config.generator.width, config.generator.height, config.generator.n_agents)
                         .wall_density(config.generator.wall_density)
                         .lasers(config.generator.n_lasers)
                         .gems(config.generator.n_gems))
            env = laser_env.GeneratorWrapper(generator)
        builder = rlenv.Builder(env)
        for wrapper in config.env_wrappers:
            match wrapper:
                case "TimeLimit": builder.time_limit(config.time_limit)
                case "IntrinsicReward": builder.intrinsic_reward("linear", initial_reward=0.5, anneal=10)
                case "AgentId": builder.agent_id()
                case other: raise ValueError(f"Unknown wrapper: {wrapper}")
        return builder.build_all()
    
    @staticmethod
    def _create_memory(config: TrainConfig) -> marl.models.ReplayMemory:
        memory_builder = marl.models.MemoryBuilder(config.memory.size, "episode" if config.recurrent else "transition")
        if config.memory.prioritized:
            memory_builder.prioritized()
        if config.memory.nstep > 1:
            memory_builder.nstep(config.memory.nstep, 0.99)
        return memory_builder.build()
        

    def train(self, params: StartTrain):
        self.runner.train(test_interval=params.test_interval, n_tests=params.num_tests, n_steps=params.num_steps, quiet=True)
        self.runner._logger._disconnect_clients = True

    def test(self, params: StartTrain):
        self.runner.train(test_interval=params.test_interval, n_tests=params.num_tests, n_steps=params.num_steps, quiet=True)
        self.runner._logger._disconnect_clients = True

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

    def get_memory_priorities(self):
        # Won't work with DQN
        match self.runner._algo.algo.algo.memory:
            case PrioritizedMemory() as pm:
                p: list[float] = []
                for i in range(len(pm._tree)):
                    p.append(pm._tree[i])
                return pm._tree.total, p
            case other: return float(len(other)), [1.] * len(other)

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
