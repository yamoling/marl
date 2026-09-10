from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from marlenv import Observation

from marl.algos.optimism import VBE
from marl.models import Action, Agent, Policy, QNetwork


@dataclass
class DQNAgent(Agent):
    """
    Deep Q-Network Interface with shared QNetwork.
    """

    qnetwork: QNetwork
    train_policy: Policy
    test_policy: Policy

    def __init__(
        self,
        qnetwork: QNetwork,
        train_policy: Policy,
        test_policy: Policy | None = None,
        vbe: VBE | None = None,
    ):
        super().__init__()
        self.qnetwork = qnetwork
        self.train_policy = train_policy
        if test_policy is None:
            test_policy = self.train_policy
        self.test_policy = test_policy
        self.policy = self.train_policy
        self.vbe = vbe
        self._is_training = True

    def choose_action(self, observation: Observation, *, with_details: bool = False):
        with torch.no_grad():
            qvalues = self.qnetwork.qvalues(observation)
        qvalues = qvalues.numpy(force=True)
        if self.qnetwork.is_multi_objective:
            qvalues = qvalues.sum(axis=-1)
        bonus = None
        if self._is_training and self.vbe is not None:
            bonus = self.vbe.compute_bonus(observation)
            qvalues = qvalues + bonus
        np_action = self.policy.get_action(qvalues, observation.available_actions)
        return Action(np_action, q_values=qvalues, vbe=bonus)

    def set_testing(self):
        self.policy = self.test_policy
        self.qnetwork.eval()
        self._is_training = False
        super().set_testing()

    def set_training(self):
        self.policy = self.train_policy
        self.qnetwork.train()
        self._is_training = True
        super().set_training()

    def save(self, to_directory: Path):
        super().save(to_directory)
        self.train_policy.to_file(to_directory / "train_policy.json")
        self.test_policy.to_file(to_directory / "test_policy.json")

    def load(self, from_directory: Path):
        super().load(from_directory)
        self.train_policy = Policy.from_file(from_directory / "train_policy.json")
        self.test_policy = Policy.from_file(from_directory / "test_policy.json")
        self.policy = self.train_policy

    def new_episode(self):
        self.qnetwork.reset_hidden_states()
