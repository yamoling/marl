import os
import shutil
import threading
import rlenv
import laser_env
from dataclasses import dataclass
from marl.models import Experiment
from marl.utils import EmptyForcedActionsException
from marl import Runner
import marl

from .messages import ExperimentConfig, TrainConfig

@dataclass
class ServerState:
    logdir: str
    loggers: dict[str, marl.logging.WSLogger]
    experiments: dict[str, Experiment]
    runners = dict[str, Runner]

    def __init__(self, logdir="logs") -> None:
        self.experiments = {}
        self.logdir = logdir
        self.loggers = {}
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
    
    def load_experiment(self, logdir: str) -> Experiment:
        if logdir in self.experiments:
            return self.experiments[logdir]
        experiment = Experiment.load(logdir)
        self.experiments[logdir] = experiment
        return experiment

    def stop_experiment(self, logdir: str):
        if logdir in self.loggers:
            logger = self.loggers.pop(logdir)
            logger.stop()
        if logdir in self.experiments:
            self.experiments.pop(logdir)
        if logdir in self.runners:
            self.runners.pop(logdir)
            

    
    def delete_experiment(self, logdir: str):
        try:
            self.stop_experiment(logdir)
            shutil.rmtree(logdir)
        except FileNotFoundError:
            raise ValueError(f"Experiment {logdir} could not be deleted !")
        
    def create_experiment(self, config: ExperimentConfig) -> Experiment:
        env = self._create_env(config)
        memory = self._create_memory(config)
        qbuilder = marl.DeepQBuilder()
        if config.vdn:
            qbuilder.vdn()
        qbuilder.qnetwork_default(env)
        qbuilder.memory(memory)
        algo = qbuilder.build()
        if config.forced_actions is not None:
            if len(config.forced_actions) == 0:
                raise EmptyForcedActionsException()
            else:
                algo = marl.wrappers.ForceActionWrapper(algo, config.forced_actions)
        algo = marl.wrappers.ReplayWrapper(algo, config.logdir)
        experiment = Experiment(
            logdir=config.logdir,
            algo=algo,
            env=env,
        )
        self.experiments[config.logdir] = experiment
        return experiment
    
    def create_runner(self, logdir: str, checkpoint_dir: str=None):
        """Creates the runner for the experiment and returns its logger"""
        if logdir not in self.runners:
            logger = marl.logging.WSLogger(logdir)
            self.loggers[logdir] = logger
            runner = self.experiments[logdir].create_runner(checkpoint_dir, logger)
            self.runners[logdir] = runner

    def get_runner(self, logdir: str, checkpoint_dir: str=None) -> Runner:
        if logdir not in self.runners:
            return self.create_runner(logdir, checkpoint_dir)
        return self.runners[logdir]
    
    def get_logger(self, logdir: str):
        return self.loggers[logdir]
    
    def train(self, logdir: str, config: TrainConfig):
        runner = self.get_runner(logdir)
        thread_function = lambda: runner.train(config.num_steps, n_tests=config.num_tests, test_interval=config.test_interval, quiet=True)
        threading.Thread(target=thread_function).start()
        return self.loggers[logdir].port

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
                case "TimePenalty": builder.time_penalty(config.time_penalty)
                case other: raise ValueError(f"Unknown wrapper: {wrapper}")
        return builder.build()
    
    @staticmethod
    def _create_memory(config: ExperimentConfig) -> marl.models.ReplayMemory:
        memory_builder = marl.models.MemoryBuilder(config.memory.size, "episode" if config.recurrent else "transition")
        if config.memory.prioritized:
            memory_builder.prioritized()
        if config.memory.nstep > 1:
            memory_builder.nstep(config.memory.nstep, 0.99)
        return memory_builder.build()