import numpy as np
from rlenv.models import EpisodeBuilder, RLEnv, Transition, Episode


def replay_video(env: RLEnv, actions: np.ndarray[np.int64]) -> list[np.ndarray[np.uint8]]:
    """Replay a sequence of actions in an environment and returns the rendered frames

    Args:
        env: The environment to replay in.
        actions: A sequence of actions to replay.
    """
    from laser_env.world import Action
    env.reset()
    frames = [env.render('rgb_array')]
    for action in actions:
        action_meanings = [Action(a) for a in action]
        print(action_meanings)
        env.step(action)
        frames.append(env.render('rgb_array'))
    return frames


def replay_episode(env: RLEnv, actions: np.ndarray[np.int64]) -> Episode:
    """Replay a sequence of actions in an environment."""
    from laser_env.world import Action

    episode = EpisodeBuilder()
    obs = env.reset()
    for action in actions:
        action_meanings = [Action(a) for a in action]
        print(action_meanings)
        obs_, reward, done, info = env.step(action)
        episode.add(Transition(obs, action, reward, done, info, obs_))
    return episode.build()

