import os
import laser_env
import rlenv
import shutil
from marl.models import Experiment, ReplayEpisodeSummary, Run, ReplayEpisode
from marl.utils import EmptyForcedActionsException
import marl

from .messages import ExperimentConfig, RunConfig, TrainConfig


class ServerState:
    def __init__(self, logdir="logs") -> None:
        self.experiments: dict[str, Experiment] = {}
        self.logdir = logdir

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
        # Reload the experiment even if it is already in memory
        experiment = Experiment.load(logdir)
        self.experiments[logdir] = experiment
        return experiment

    def unload_experiment(self, logdir: str) -> Experiment | None:
        return self.experiments.pop(logdir, None)

    def delete_experiment(self, logdir: str):
        try:
            experiment = self.unload_experiment(logdir)
            if experiment is None:
                experiment = Experiment.load(logdir)
            for run in experiment.runs:
                run.stop()
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
        experiment = Experiment.create(config.logdir, env, algo, 300_000)
        self.experiments[config.logdir] = experiment
        return experiment

    def create_runner(self, logdir: str, run_config: RunConfig):
        """Creates a runner for the given experiment and returns their loggers"""
        if logdir not in self.experiments:
            raise ValueError(f"Experiment {logdir} not found")
        experiment = self.experiments[logdir]
        raise NotImplementedError()

    def stop_runner(self, rundir: str):
        logdir = os.path.dirname(rundir)
        if logdir not in self.experiments:
            raise ValueError(f"Experiment {rundir} not found")
        self.experiments[logdir].stop_runner(rundir)

    def restart_runner(self, rundir: str, train_config: TrainConfig):
        logdir = os.path.dirname(rundir)
        if logdir not in self.experiments:
            self.experiments[logdir] = Experiment.load(logdir)
        raise NotImplementedError()

    def delete_runner(self, rundir: str):
        shutil.rmtree(rundir)

    def get_test_episodes_at(
        self, logdir: str, time_step: int
    ) -> list[ReplayEpisodeSummary]:
        if logdir not in self.experiments:
            experiment_dir = Experiment.find_experiment_directory(logdir)
            self.load_experiment(experiment_dir)
        res = self.experiments[logdir].get_test_episodes(time_step)
        return res

    def get_runner_port(self, rundir: str) -> int:
        run = Run.load(rundir)
        return run.get_port()

    def replay_episode(self, episode_dir: str) -> ReplayEpisode:
        longest_match = ""
        matching_experiment = None
        for logdir, experiment in self.experiments.items():
            if episode_dir.startswith(logdir) and len(logdir) > len(longest_match):
                longest_match = logdir
                matching_experiment = experiment
        if matching_experiment is None:
            raise ValueError(f"Could not find experiment for episode {episode_dir}")
        return matching_experiment.replay_episode(episode_dir)

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
                case "TimeLimit":
                    builder.time_limit(config.time_limit)
                case "IntrinsicReward":
                    builder.intrinsic_reward("linear", initial_reward=0.5, anneal=10)
                case "AgentId":
                    builder.agent_id()
                case "TimePenalty":
                    builder.time_penalty(config.time_penalty)
                case other:
                    raise ValueError(f"Unknown wrapper: {wrapper}")
        return builder.build()

    @staticmethod
    def _create_memory(config: ExperimentConfig) -> marl.models.ReplayMemory:
        memory_builder = marl.models.MemoryBuilder(
            config.memory.size, "episode" if config.recurrent else "transition"
        )
        if config.memory.prioritized:
            memory_builder.prioritized()
        if config.memory.nstep > 1:
            memory_builder.nstep(config.memory.nstep, 0.99)
        return memory_builder.build()


def _start_process_function(experiment: Experiment, run_config: RunConfig):
    # os.setpgrp is used to prevent CTRL+C from killing the child process (but still terminate the server)
    os.setpgrp()
    runner = experiment.create_runner(seed=run_config.seed)
    runner.train(n_tests=run_config.num_tests, quiet=True)


def _restart_process_function(
    experiment: Experiment, rundir: str, train_config: TrainConfig
):
    # os.setpgrp is used to prevent CTRL+C from killing the child process (but still terminate the server)
    os.setpgrp()
    runner = experiment.restore_runner(rundir)
    runner.train(
        n_steps=train_config.num_steps,
        n_tests=train_config.num_tests,
        test_interval=train_config.test_interval,
        quiet=True,
    )
