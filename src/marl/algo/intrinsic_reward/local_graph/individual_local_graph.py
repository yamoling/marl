import numpy as np
from marlenv import MARLEnv
from marlenv.models.env import ActionSpaceType
from marl.models.trainer import Trainer
from dataclasses import dataclass

from .local_graph import LocalGraphBottleneckFinder


@dataclass
class IndividualLocalGraphTrainer(Trainer):
    agent_dims: int
    n_agents: int
    local_graphs: list[LocalGraphBottleneckFinder[np.ndarray]]

    def __init__(self, env: MARLEnv[ActionSpaceType]):
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
                agent_state: np.ndarray = state[i * self.agent_dims : (i + 1) * self.agent_dims]
                agent_states[i].append(tuple(agent_state.tolist()))
        for graph, states in zip(self.local_graphs, agent_states):
            graph.add_trajectory(states)
        return logs

    def to(self, _):
        return self

    def randomize(self):
        return
