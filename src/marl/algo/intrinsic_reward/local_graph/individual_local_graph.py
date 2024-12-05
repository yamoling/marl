import numpy as np
from marlenv import MARLEnv
from marl.models.trainer import Trainer

from .local_graph import LocalGraphBottleneckFinder


class IndividualLocalGraphTrainer(Trainer):
    def __init__(self, env: MARLEnv):
        super().__init__("episode")
        self.agent_dims = env.agent_state_size
        self.n_agents = env.n_agents
        self.local_graphs = [LocalGraphBottleneckFinder[np.ndarray]() for _ in range(self.n_agents)]

    def update_episode(self, episode, episode_num, time_step):
        # Access the private _observations attribute of the episode
        # that contains all the observations including the start and end
        # observations of the episode.
        logs = {}
        agent_states = [[] for _ in range(self.n_agents)]
        for state in episode._states:
            for i in range(self.n_agents):
                agent_states.append(state[i * self.agent_dims : (i + 1) * self.agent_dims])
        for graph, states in zip(self.local_graphs, agent_states):
            graph._build_local_graph(states)
        return logs

    def to(self, _):
        return self

    def randomize(self):
        return
