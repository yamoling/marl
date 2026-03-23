from marlenv import Observation
import numpy as np
from marl.models import Policy
from marl.models.agent import Agent
from dataclasses import dataclass


@dataclass
class QAgent(Agent):
    train_policy: Policy
    test_policy: Policy

    def __init__(self, qtable: dict[Observation, np.ndarray], policy: Policy, test_policy: Policy | None):
        super().__init__()
        self._qtable = qtable
        self.train_policy = policy
        if test_policy is None:
            test_policy = policy
        self.test_policy = test_policy
        self.policy = policy

    def set_testing(self):
        super().set_testing()
        self.policy = self.test_policy

    def set_training(self):
        super().set_training()
        self.policy = self.train_policy

    def choose_action(self, observation: Observation):
        qvalues = self._qtable[observation].copy()
        return self.policy.get_action(qvalues, observation.available_actions)
