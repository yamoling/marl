import marl
from .utils import generate_episode, MockEnv


def test_episode_batch_padded():
    from rlenv.wrappers import TimeLimit

    env = TimeLimit(MockEnv(2), 5)
    episodes = []
    for i in range(5, 10):
        env.step_limit = i
        episodes.append(generate_episode(env))
    batch = marl.models.batch.EpisodeBatch(episodes, list(range(len(episodes))))
    assert len(batch.obs) == 9
    assert len(batch.obs_) == 10
    assert len(batch.masks) == 9
