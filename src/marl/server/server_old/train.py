from dataclasses import dataclass
import marl
import rlenv
from copy import deepcopy
import laser_env
from rlenv import Transition
from marl.models import PrioritizedMemory
from marl.utils.others import encode_b64_image
from marl import Runner

from .messages import ExperimentConfig, TrainConfig


@dataclass
class TrainServerState:
    runner: marl.Runner | None
    env: rlenv.RLEnv | None
    
    def __init__(self) -> None:
        self.runner = None

    def create_runner(self, config: ExperimentConfig):
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
    
    def load_checkpoint(self, checkpoint_dir: str):
        self.runner = Runner.from_checkpoint(checkpoint_dir)

    @staticmethod
    def _create_env(config: ExperimentConfig):
        obs_type = laser_env.ObservationType.from_str(config.obs_type)
        if config.static_map:
            env = laser_env.StaticLaserEnv(config.level, obs_type)
        else:
            env = laser_env.DynamicLaserEnv(
                width=config.generator.width, 
                height=config.generator.height, 
                num_agents=config.generator.n_agents,
                num_gems=config.generator.n_gems,
                num_lasers=config.generator.n_lasers,
                obs_type=obs_type,
                wall_density=config.generator.wall_density,
            )
        builder = rlenv.Builder(env)
        for wrapper in config.env_wrappers:
            match wrapper:
                case "TimeLimit": builder.time_limit(config.time_limit)
                case "IntrinsicReward": builder.intrinsic_reward("linear", initial_reward=0.5, anneal=10)
                case "AgentId": builder.agent_id()
                case other: raise ValueError(f"Unknown wrapper: {wrapper}")
        return builder.build_all()
    
    @staticmethod
    def _create_memory(config: ExperimentConfig) -> marl.models.ReplayMemory:
        memory_builder = marl.models.MemoryBuilder(config.memory.size, "episode" if config.recurrent else "transition")
        if config.memory.prioritized:
            memory_builder.prioritized()
        if config.memory.nstep > 1:
            memory_builder.nstep(config.memory.nstep, 0.99)
        return memory_builder.build()

    def train(self, params: TrainConfig):
        self.runner.train(test_interval=params.test_interval, n_tests=params.num_tests, n_steps=params.num_steps, quiet=True)
        self.runner._logger._disconnect_clients = True

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
