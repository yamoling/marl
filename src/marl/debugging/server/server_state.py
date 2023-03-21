import os
import shutil
import rlenv
import laser_env
from dataclasses import dataclass
from marl.models import Experiment
from marl import Runner
import marl

from .messages import TrainConfig, StartTrain

@dataclass
class ServerState:
    runners: dict[str, Runner]

    def __init__(self, logdir="logs") -> None:
        self.logdir = logdir
        self.runners = {}

    def list_experiments(self):
        experiments: dict[str, Experiment] = {}
        for directory in os.listdir(self.logdir):
            directory = os.path.join(self.logdir, directory)
            try:
                experiments[directory] = Experiment.load(directory)
            except FileNotFoundError:
                # Not an experiment directory, ignore
                pass
        return experiments
    
    def stop_experiment(self, logdir: str):
        raise NotImplementedError()
    
    def delete_experiment(self, logdir: str):
        try:
            shutil.rmtree(logdir)
        except FileNotFoundError:
            raise ValueError(f"Experiment {logdir} could not be deleted !")

    def start_exeperiment(self, config: TrainConfig):
        logger = marl.logging.WSLogger(config.logdir)
        env, test_env = self._create_env(config)
        memory = self._create_memory(config)
        qbuilder = marl.DeepQBuilder()
        if config.vdn:
            qbuilder.vdn()
        qbuilder.qnetwork_default(env)
        qbuilder.memory(memory)
        algo = marl.wrappers.ReplayWrapper(qbuilder.build(), logger.logdir)
        runner = marl.Runner(env, test_env=test_env, algo=algo, logger=logger)
        self.runners[runner.logdir] = runner
        return logger.port
    
    def train(self, params: StartTrain):
        self.runner.train(test_interval=params.test_interval, n_tests=params.num_tests, n_steps=params.num_steps, quiet=True)
        self.runner._logger._disconnect_clients = True

    @staticmethod
    def _create_env(config: TrainConfig):
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
    def _create_memory(config: TrainConfig) -> marl.models.ReplayMemory:
        memory_builder = marl.models.MemoryBuilder(config.memory.size, "episode" if config.recurrent else "transition")
        if config.memory.prioritized:
            memory_builder.prioritized()
        if config.memory.nstep > 1:
            memory_builder.nstep(config.memory.nstep, 0.99)
        return memory_builder.build()