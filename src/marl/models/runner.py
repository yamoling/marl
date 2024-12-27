from copy import deepcopy
from typing import Literal, Optional
from marlenv import Episode, MARLEnv, Transition, ActionSpace
import torch
import numpy as np
from tqdm import tqdm
from marl.agents import Agent
from marl.models.run import Run, RunHandle
from marl.utils import get_device
from marl.agents.random_algo import RandomAgent
from marl.models.trainer import Trainer
from marl.training import NoTrain


from typing_extensions import TypeVar


A = TypeVar("A", bound=ActionSpace)


class Runner[A, AS: ActionSpace]:
    _env: MARLEnv[A, AS]
    _algo: Agent
    _trainer: Trainer
    _test_env: MARLEnv[A, AS]

    def __init__(
        self,
        env: MARLEnv[A, AS],
        agent: Optional[Agent] = None,
        trainer: Optional[Trainer] = None,
        test_env: Optional[MARLEnv[A, AS]] = None,
    ):
        self._trainer = trainer or NoTrain(env)
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
        render_tests: bool,
    ):
        self._env.seed(step_num)
        obs, state = self._env.reset()
        self._algo.new_episode()
        episode = Episode.new(obs, state)
        episode.add_metrics({"initial_value": self._algo.value(obs)})
        while not episode.is_finished and step_num < max_step:
            step_num += 1
            if n_tests > 0 and test_interval > 0 and step_num % test_interval == 0:
                self.test(n_tests, step_num, quiet, run_handle, render_tests)
            match self._algo.choose_action(obs):
                case (action, dict(kwargs)):
                    step = self._env.step(action)
                case action:
                    step = self._env.step(action)
                    kwargs = {}
            if step_num == max_step:
                step.truncated = True
            transition = Transition.from_step(obs, state, action, step, **kwargs)
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
        render_tests: bool = False,
    ):
        """Start the training loop"""
        import torch
        import random
        import numpy as np

        # The environment is seeded at each training and testing step for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.randomize()

        max_step = n_steps
        episode_num = 0
        step = 0
        pbar = tqdm(total=n_steps, desc="Training", unit="Step", leave=True, disable=quiet)
        with Run.create(logdir, seed) as run:
            if n_tests > 0 and test_interval > 0:
                self.test(n_tests, 0, quiet, run, render_tests)
            while step < max_step:
                episode = self._train_episode(
                    step_num=step,
                    episode_num=episode_num,
                    n_tests=n_tests,
                    quiet=quiet,
                    run_handle=run,
                    max_step=max_step,
                    test_interval=test_interval,
                    render_tests=render_tests,
                )
                episode_num += 1
                step += len(episode)
                pbar.update(len(episode))
        pbar.close()

    def test(self, n_tests: int, time_step: int, quiet: bool, run_handle: RunHandle, render: bool):
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
                if render:
                    self._test_env.render()
                match self._algo.choose_action(obs):
                    case (action, _):
                        step = self._test_env.step(action)
                    case action:
                        step = self._test_env.step(action)
                transition = Transition.from_step(obs, state, action, step)
                episode.add(transition)
                obs = step.obs
                state = step.state
            if render:
                self._test_env.render()
            episodes.append(episode)
        run_handle.log_tests(episodes, self._algo, time_step)
        if not quiet:
            avg_score = np.sum([e.score for e in episodes], axis=0) / n_tests
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
