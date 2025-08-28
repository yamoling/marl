from typing import Sequence
from marlenv import RLEnvWrapper, MARLEnv, Space, State
import numpy as np
from dataclasses import dataclass


@dataclass
class StateCounter[S: Space](RLEnvWrapper[S]):
    def __init__(self, wrapped: MARLEnv[S]):
        super().__init__(wrapped)
        self._per_agent = [set[int]() for _ in range(self.n_agents)]
        self._joint = set[int]()

    def _register(self, state: State):
        self._joint.add(hash(state))
        eh = hash(state.extras.tobytes())
        for i in range(self.n_agents):
            agent_data = state.data[i * self.agent_state_size : (i + 1) * self.agent_state_size]
            h = hash((agent_data.tobytes(), eh))
            self._per_agent[i].add(h)

    def step(self, actions: np.ndarray | Sequence):
        step = super().step(actions)
        self._register(step.state)
        if step.is_terminal:
            step.info = step.info | {
                "joint-count": len(self._joint),
                **{f"agent-{i}-count": len(agent_set) for i, agent_set in enumerate(self._per_agent)},
            }
        return step

    def reset(self):
        obs, state = super().reset()
        self._register(state)
        return obs, state
