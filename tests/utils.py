from rlenv.models import RLEnv, Episode, Transition, EpisodeBuilder


def generate_episode(env: RLEnv) -> Episode:
    obs = env.reset()
    episode = EpisodeBuilder()
    while not episode.is_finished:
        action = env.action_space.sample()
        next_obs, r, done, truncated, info = env.step(action)
        episode.add(Transition(obs, action, r, done, info, next_obs, truncated))
        obs = next_obs
    return episode.build()



import numpy as np
from rlenv import RLEnv, Observation
from rlenv.models import DiscreteActionSpace


class MockEnv(RLEnv[DiscreteActionSpace]):
    OBS_SIZE = 42
    N_ACTIONS = 5
    END_GAME = 30
    REWARD_STEP = 1

    def __init__(self, n_agents) -> None:
        super().__init__(DiscreteActionSpace(n_agents, MockEnv.N_ACTIONS))
        self._n_agents = n_agents
        self.t = 0
        self.actions_history = []

    @property
    def observation_shape(self):
        return (MockEnv.OBS_SIZE,)

    @property
    def state_shape(self):
        return (0,)

    def kwargs(self) -> dict[str,]:
        return {"n_agents": self.n_agents}

    def reset(self):
        self.t = 0
        return self.observation()

    def observation(self):
        obs_data = np.array([np.arange(self.t + agent, self.t + agent + MockEnv.OBS_SIZE) for agent in range(self.n_agents)])
        return Observation(obs_data, self.get_avail_actions(), self.get_state())

    def get_state(self):
        return np.array([])

    def render(self, mode: str = "human"):
        return

    def step(self, action):
        self.t += 1
        self.actions_history.append(action)
        return self.observation(), MockEnv.REWARD_STEP, self.t >= MockEnv.END_GAME, False, {}
