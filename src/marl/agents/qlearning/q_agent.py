from marlenv import Observation
import numpy as np
from marl.models import Policy
from ..agent import Agent


class QAgent(Agent):
    def __init__(self, qtable: dict, policy: Policy, test_policy: Policy | None):
        super().__init__()
        self.qtable = qtable
        self.train_policy = policy
        if test_policy is None:
            test_policy = policy
        self.test_policy = test_policy
        self.policy = policy

    def choose_action(self, observation: Observation):
        qvalues = self.compute_qvalues(observation)
        return self.policy.get_action(qvalues, observation.available_actions)

    def compute_qvalues(self, obs: Observation):
        qvalues = []
        obs_data = np.concatenate((obs.data, obs.extras), axis=-1)
        for agent_obs in obs_data:
            h = self.hash_ndarray(agent_obs)
            if h not in self.qtable:
                self.qtable[h] = np.ones(obs.available_actions.shape[-1], dtype=np.float32)
            agent_qvalues = self.qtable[h]
            qvalues.append(agent_qvalues)
        return np.array(qvalues)

    @staticmethod
    def hash_ndarray(data: np.ndarray) -> int:
        return hash(data.tobytes())
