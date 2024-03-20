import torch
import marl
import random
from rlenv import Builder, Transition
from rlenv.models import EpisodeBuilder

from rlenv import MockEnv


if __name__ == "__main__":
    env = Builder(MockEnv(2)).time_limit(4).build()
    done = truncated = False
    obs = env.reset()
    episode = EpisodeBuilder()
    while not (done or truncated):
        actions = env.action_space.sample(obs.available_actions)
        obs_, r, done, truncated, info = env.step(actions)
        episode.add(Transition(obs, actions, r, done, info, obs_, truncated))
        obs = obs_

    episode = episode.build()
    transitions = list(episode.transitions())
    # print(transitions)
    random.shuffle(transitions)
    # batch = marl.models.batch.TransitionBatch(transitions)
    batch = marl.models.batch.EpisodeBatch([episode, episode, episode])

    print(batch.obs[:, 1])
    print(batch.obs_[:, 1])
    print(batch.all_obs[:, 1])

    for i in range(len(batch)):
        assert torch.equal(batch.obs[i], batch.all_obs[i])
        assert torch.equal(batch.obs_[i], batch.all_obs[i + 1])
