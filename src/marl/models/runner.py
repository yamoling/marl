from copy import deepcopy
from typing import Literal, Optional
from marlenv import Episode, EpisodeBuilder, MARLEnv, Transition, ActionSpace
import torch
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
import marl
from marl.agents import Agent
from marl.models.run import Run, RunHandle
from marl.utils import get_device
from marl.agents.random_algo import RandomAgent
from marl.models.trainer import Trainer
from marl.training import NoTrain


from typing import Generic
from typing_extensions import TypeVar

A = TypeVar("A", bound=ActionSpace)
O = TypeVar("O", bound=npt.NDArray[np.float32])  # noqa: E741 (Ambiguous variable name 'O')
S = TypeVar("S", bound=npt.NDArray[np.float32])
R = TypeVar("R", bound=npt.NDArray[np.float32] | float)


class Runner(Generic[A, O, S, R]):
    env: MARLEnv[A, O, S, R]
    algo: Agent[O]
    trainer: Trainer
    test_env: MARLEnv[A, O, S, R]

    def __init__(
        self,
        env: MARLEnv[A, O, S, R],
        agent: Optional[Agent[O]] = None,
        trainer: Optional[Trainer] = None,
        test_env: Optional[MARLEnv[A, O, S, R]] = None,
    ):
        self._trainer = trainer or NoTrain()
        self._env = env
        if agent is None:
            agent = RandomAgent(env)
        self._algo = agent  #  or RandomAlgo(env)
        if test_env is None:
            test_env = deepcopy(env)
        self._test_env = test_env

    def _train_episode(
        self,
        step_num: int,
        episode_num: int,
        n_tests: int,
        quiet: bool,
        run_handle: RunHandle,
        max_step: int,
        test_interval: int,
    ):
        self._env.seed(step_num)
        obs, state = self._env.reset()
        self._algo.new_episode()
        episode = Episode.new(obs, state)
        episode.add_metrics({"initial_value": self._algo.value(obs)})
        while not episode.is_finished and step_num < max_step:
            step_num += 1
            if n_tests > 0 and test_interval > 0 and step_num % test_interval == 0:
                self.test(n_tests, step_num, quiet, run_handle)
            action = self._algo.choose_action(obs)
            step = self._env.step(action)
            if step_num == max_step:
                step.truncated = True
            transition = Transition.from_step(obs, state, action, step)
            training_metrics = self._trainer.update_step(transition, step_num)
            run_handle.log_train_step(training_metrics, step_num)
            episode.add(transition)
            obs = step.obs
            state = step.state
        training_logs = self._trainer.update_episode(episode, episode_num, step_num)
        run_handle.log_train_episode(episode, step_num, training_logs)
        return episode

    def run(
        self,
        logdir: str,
        seed: int = 0,
        n_tests: int = 1,
        n_steps: int = 1_000_000,
        test_interval: int = 5_000,
        quiet: bool = False,
    ):
        """Start the training loop"""
        marl.seed(seed, self._env)
        self.randomize()
        max_step = n_steps
        episode_num = 0
        step = 0
        pbar = tqdm(total=n_steps, desc="Training", unit="Step", leave=True, disable=quiet)
        with Run.create(logdir, seed) as run:
            if n_tests > 0 and test_interval > 0:
                self.test(n_tests, 0, quiet, run)
            while step < max_step:
                episode = self._train_episode(
                    step_num=step,
                    episode_num=episode_num,
                    n_tests=n_tests,
                    quiet=quiet,
                    run_handle=run,
                    max_step=max_step,
                    test_interval=test_interval,
                )
                episode_num += 1
                step += len(episode)
                pbar.update(len(episode))
        pbar.close()

    def test(self, n_tests: int, time_step: int, quiet: bool, run_handle: RunHandle):
        """Test the agent"""
        self._algo.set_testing()
        episodes = list[Episode]()
        for test_num in tqdm(range(n_tests), desc="Testing", unit="Episode", leave=True, disable=quiet):
            self._test_env.seed(time_step + test_num)
            obs, state = self._test_env.reset()
            episode = Episode.new(obs, state)
            self._algo.new_episode()
            episode.add_metrics({"initial_value": self._algo.value(obs)})
            i = 0
            while not episode.is_finished:
                i += 1
                action = self._algo.choose_action(obs)
                step = self._test_env.step(action)
                transition = Transition.from_step(obs, state, action, step)
                episode.add(transition)
                obs = step.obs
                state = step.state
            episodes.append(episode)
        run_handle.log_tests(episodes, self._algo, time_step)
        if not quiet:
            avg_score = sum(e.score for e in episodes) / n_tests
            print(f"{time_step:9d} Average score: {avg_score}")
        self._algo.set_training()

    def to(self, device: Literal["auto", "cpu"] | int | torch.device):
        match device:
            case str():
                device = get_device(device)
            case int():
                device = torch.device(device)
        self._algo.to(device)
        self._trainer.to(device)
        return self

    def randomize(self):
        self._algo.randomize()
        self._trainer.randomize()
