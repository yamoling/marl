import logging
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch
from marlenv import Episode, MARLEnv, Transition
from tqdm import tqdm

if TYPE_CHECKING:
    from marl import Agent, Run, Trainer


def simple_run[E: MARLEnv, T: npt.ArrayLike](run: "Run[E, T]", quiet: bool, render_tests: bool, device: torch.device):
    """
    Boilerplate to run an RL experiment:
        - Seeding first
        - Creation of the trainer, agent, logger and training and testing environments
        - Train the agent
        - Tests **exactly** every `run.test_interval` steps
        - Log training and testing data
    """
    with tqdm(total=run.n_steps, desc="Training", unit="step", leave=True, disable=quiet) as pbar, run:
        import marl

        env, test_env = run.env.make(), run.test_env.make()
        trainer = run.trainer.to(device)
        agent = trainer.make_agent().to(device)
        marl.seed(run.seed, env, test_env)
        trainer.randomize()
        agent.randomize()

        episode_num, time_step = 0, 0
        while time_step < run.n_steps:
            episode = _train_episode(env, test_env, agent, trainer, time_step, episode_num, render_tests, quiet, run)
            episode_num += 1
            time_step += len(episode)
            pbar.update(len(episode))
        # Test the final agent
        if run.should_test_at(time_step):
            _test_and_log(test_env, agent, time_step, render_tests, quiet, run)


def _train_episode[A](
    env: MARLEnv[A],
    test_env: MARLEnv[A],
    agent: "Agent[A]",
    trainer: "Trainer[A]",
    time_step: int,
    episode_num: int,
    render_tests: bool,
    quiet: bool,
    run: "Run",
):
    obs, state = env.reset()
    agent.new_episode()
    episode = Episode.new(obs, state, metrics={"episode_num": episode_num})
    while not episode.is_finished:
        if run.should_test_at(time_step):
            _test_and_log(test_env, agent, time_step, render_tests, quiet, run)
        action = agent.choose_action(obs)
        step = env.step(action)
        if time_step == run.n_steps:
            step.truncated = True
        transition = Transition.from_step(obs, state, action.action, step, **action.details)
        training_metrics = trainer.update_step(transition, time_step)
        run.logger.log_training_data(training_metrics, time_step)
        episode.add(transition)
        obs = step.obs
        state = step.state
        time_step += 1
    run.logger.log_train(episode.metrics, time_step)
    training_logs = trainer.update_episode(episode, episode_num, time_step)
    run.logger.log_training_data(training_logs, time_step)
    return episode


def _test_and_log[A](test_env: MARLEnv[A], agent: "Agent[A]", time_step: int, render: bool, quiet: bool, run: "Run"):
    if run.save_weights:
        run.logger.save_agent(agent, time_step)
    agent.set_testing()
    episodes = list[Episode]()
    for test_num in tqdm(range(run.n_tests), desc="Testing", unit="Episode", leave=True, disable=quiet):
        seed = get_test_seed(time_step, test_num)
        episodes.append(seeded_rollout(test_env, agent, seed, render, compute_frames=False)[0])
    if not quiet:
        metrics = episodes[0].metrics.keys()
        avg_metrics = {}
        for key in metrics:
            try:
                avg_metrics[key] = sum([e.metrics[key] for e in episodes]) / run.n_tests
            except TypeError:
                pass
        logging.info(avg_metrics)
    run.logger.log_test_episodes(episodes, time_step, run.save_actions)
    agent.set_training()


def seeded_rollout[A](env: MARLEnv[A], agent: "Agent[A]", seed: int, render=False, compute_frames=False):
    agent.set_testing()
    env.seed(seed)
    agent.seed(seed)

    agent.new_episode()
    obs, state = env.reset()
    episode = Episode.new(obs, state)
    frames = list[npt.NDArray[np.uint8]]()
    action_details = []
    while not episode.is_finished:
        if render:
            env.render()
        if compute_frames:
            frames.append(env.get_image())
        action = agent.choose_action(obs, with_details=True)
        action_details.append(action)
        step = env.step(action.action)
        transition = Transition.from_step(obs, state, action.action, step)
        episode.add(transition)
        obs = step.obs
        state = step.state
    if render:
        env.render()
    if compute_frames:
        frames.append(env.get_image())
    return episode, frames, action_details


def get_test_seed(time_step: int, test_num: int):
    return time_step * 31 + test_num
