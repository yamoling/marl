import torch
import marl
import numpy as np
import random
from rlenv import Builder, Transition, MockEnv, EpisodeBuilder, RLEnv
from marl.models.batch import TransitionBatch


def generate_episode(env: RLEnv):
    obs = env.reset()
    episode = EpisodeBuilder()
    while not episode.is_finished:
        action = env.action_space.sample()
        next_obs, r, done, truncated, info = env.step(action)
        episode.add(Transition(obs, action, r, done, info, next_obs, truncated))
        obs = next_obs
    return episode.build()


if __name__ == "__main__":
    MockEnv.OBS_SIZE = 1
    env = Builder(MockEnv(1)).time_limit(4).build()
    episode = generate_episode(env)
    transitions = list(episode.transitions())
    random.shuffle(transitions)
    batch = TransitionBatch(transitions)

    print(batch.all_obs_)

    for i, t in enumerate(transitions):
        print(i)
        print("obs=", t.obs.data.squeeze())
        print("obs_=", t.obs_.data.squeeze())
        print("batch.obs=", batch.obs[i].squeeze())
        print("batch.obs_=", batch.obs_[i].squeeze())
        print()

    obs = torch.from_numpy(np.array([t.obs.data for t in transitions]))
    obs_ = torch.from_numpy(np.array([t.obs_.data for t in transitions]))
    print(obs)
    print(batch.all_obs_)
    print(batch.obs)
    assert torch.equal(batch.obs, obs)
    assert torch.equal(batch.obs_, obs_)
    assert torch.equal(batch.all_obs_[1:], obs_)
