import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import orjson
from marlenv import MARLEnv
from marlenv.models.env import ActionSpaceType

from marl.models.trainer import Trainer
from marl.utils import default_serialization

from .local_graph import LocalGraphBottleneckFinder


@dataclass
class IndividualLocalGraphTrainer(Trainer):
    agent_dims: int
    n_agents: int
    local_graphs: list[LocalGraphBottleneckFinder[np.ndarray]]
    update_after_n_states: int

    def __init__(self, env: MARLEnv[ActionSpaceType], update_after_n_states: int, logdir: str):
        super().__init__("episode")
        self.agent_dims = env.agent_state_size
        self.n_agents = env.n_agents
        self.local_graphs = [LocalGraphBottleneckFinder[np.ndarray]() for _ in range(self.n_agents)]
        self.n_states_visited = 0
        self.update_after_n_states = update_after_n_states
        self.logdir = os.path.join(logdir, "bottlenecks")

    def update_episode(self, episode, episode_num, time_step):
        # Access the private _observations attribute of the episode
        # that contains all the observations including the start and end
        # observations of the episode.
        logs = {}
        agent_states = [[] for _ in range(self.n_agents)]
        for state in episode._states:
            for i in range(self.n_agents):
                agent_state = state[i * self.agent_dims : (i + 1) * self.agent_dims]
                agent_states[i].append(tuple(agent_state.tolist()))
        for graph, states in zip(self.local_graphs, agent_states):
            graph.add_trajectory(states)
        self.n_states_visited += len(episode)
        if self.n_states_visited >= self.update_after_n_states:
            for agent, graph in enumerate(self.local_graphs):
                b, _ = graph.find_bottleneck()
                directory = os.path.join(self.logdir, str(time_step), datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
                os.makedirs(directory, exist_ok=True)
                with open(os.path.join(directory, f"bottlenecks-agent={agent}.json"), "wb") as f:
                    f.write(orjson.dumps(b, default=default_serialization))
                by_edge, by_vertex = graph.predict_all()
                with open(os.path.join(directory, f"edge-predictions-agent={agent}.json"), "wb") as f:
                    f.write(orjson.dumps(list(by_edge.items())))
                with open(os.path.join(directory, f"vertex-predictions-agent={agent}.json"), "wb") as f:
                    f.write(orjson.dumps(list(by_vertex.items())))
                # plt = draw_graph(graph.local_graph, b, labels)
                # plt.savefig(f"local_graph-t={time_step}-agent={agent}.png")
                graph.clear()
            self.n_states_visited = 0
        return logs

    def to(self, _):
        return self

    def randomize(self):
        return
