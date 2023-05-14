import numpy as np
from rlenv import Observation
from marl.models.algo import RLAlgo


class RandomAgent(RLAlgo):
    def __init__(self, n_actions: int, n_agents: int):
        super().__init__()
        self._n_actions = n_actions
        self._n_agents = n_agents

    def choose_action(self, observation: Observation) -> np.ndarray[np.int64]:
        actions = []
        for available in observation.available_actions:
            probs = available / available.sum()
            actions.append(np.random.choice(self._n_actions, p=probs))
        return actions
        
    def save(self, to_path: str):
        return
    
    def load(self, from_path: str):
        return
    
    def summary(self) -> dict:
        return {
            **super().summary(),
            "n_actions": self._n_actions,
            "n_agents": self._n_agents
        }

    @classmethod
    def from_summary(cls, summary: dict):
        return cls(summary["n_actions"], summary["n_agents"])
