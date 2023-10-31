import rlenv
import marl
from .utils import generate_episode, MockEnv


def test_episode_batch_padded():
    env = rlenv.wrappers.TimeLimitWrapper(MockEnv(2), 5)
    episodes = []
    for i in range(5, 10):
        env._step_limit = i
        episodes.append(generate_episode(env))
    batch = marl.models.batch.EpisodeBatch(episodes, range(len(episodes)))
    assert len(batch.obs) == 9
    assert len(batch.obs_) == 10
    assert len(batch.masks) == 9