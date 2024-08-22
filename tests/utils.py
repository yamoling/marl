from marlenv import RLEnv, Episode, Transition, EpisodeBuilder

import torch


def almost_equal(a, b, eps=1e-5):
    return abs(a - b) < eps


def parameters_equal(p1: list[torch.nn.Parameter], p2: list[torch.nn.Parameter]) -> bool:
    for (
        a,
        b,
    ) in zip(p1, p2):
        if not torch.equal(a, b):
            return False
    return True


def generate_episode(env: RLEnv):
    obs = env.reset()
    episode = EpisodeBuilder()
    while not episode.is_finished:
        action = env.action_space.sample()
        next_obs, r, done, truncated, info = env.step(action)
        episode.add(Transition(obs, action, r, done, info, next_obs, truncated))
        obs = next_obs
    return episode.build()
