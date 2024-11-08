from copy import deepcopy
from typing import Literal, Optional
from marlenv import Episode, EpisodeBuilder, MARLEnv, Transition
import torch
from tqdm import tqdm
import marl
from marl.algo import RLAlgo, DDPG
from marl.models.run import Run, RunHandle
from marl.utils import get_device
from marl.algo.random_algo import RandomAlgo
from marl.training import Trainer, NoTrain


class Runner:
    def __init__(
        self,
        env: MARLEnv,
        algo: Optional[RLAlgo] = None,
        trainer: Optional[Trainer] = None,
        test_env: Optional[MARLEnv] = None,
    ):
        self._trainer = trainer or NoTrain()
        self._env = env
        self._algo = algo or RandomAlgo(env)
        if test_env is None:
            test_env = deepcopy(env)
        self._test_env = test_env

    def _train_episode(
        self, step_num: int, episode_num: int, n_tests: int, quiet: bool, run_handle: RunHandle, max_step: int, test_interval: int
    ):
        episode = EpisodeBuilder()
        self._env.seed(step_num)
        obs = self._env.reset()
        self._algo.new_episode()
        initial_value = self._algo.value(obs)
        while not episode.is_finished and step_num < max_step:
            step_num += 1
            if n_tests > 0 and test_interval > 0 and step_num % test_interval == 0:
                self.test(n_tests, step_num, quiet, run_handle)
            action = self._algo.choose_action(obs)
            obs_, reward, done, truncated, info = self._env.step(action)
            if step_num == max_step:
                truncated = True
            if isinstance(self._algo, DDPG):  # needs old probs because off policy training
                logits = self._algo.actions_logits(obs)
                probs = torch.distributions.Categorical(logits=logits).probs.cpu().detach().numpy()  # type: ignore
                transition = Transition(obs, action, reward, done, info, obs_, truncated, probs)  # type: ignore
            else:
                transition = Transition(obs, action, reward, done, info, obs_, truncated)

            training_metrics = self._trainer.update_step(transition, step_num)
            run_handle.log_train_step(training_metrics, step_num)
            episode.add(transition)
            obs = obs_
        episode = episode.build({"initial_value": initial_value})
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
            episode = EpisodeBuilder()
            obs = self._test_env.reset()
            self._algo.new_episode()
            intial_value = self._algo.value(obs)
            i = 0
            while not episode.is_finished:
                i += 1
                action = self._algo.choose_action(obs)
                new_obs, reward, done, truncated, info = self._test_env.step(action)
                transition = Transition(obs, action, reward, done, info, new_obs, truncated)
                episode.add(transition)
                obs = new_obs
            episode = episode.build({"initial_value": intial_value})
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
