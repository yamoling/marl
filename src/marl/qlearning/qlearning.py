from abc import abstractmethod
import torch
from marl.policy import Policy
from marl.models import ReplayMemory, Batch, RLAlgo
from rlenv import Observation


class IQLearning(RLAlgo):
    @property
    @abstractmethod
    def gamma(self) -> float:
        """The discount factor"""

    @property
    @abstractmethod
    def policy(self) -> Policy:
        """The qlearning policy"""

    @abstractmethod
    def compute_qvalues(self, data: Observation) -> torch.Tensor:
        """Compute the qvalues for the given input data."""


class IDeepQLearning(IQLearning):
    @abstractmethod
    def compute_qvalues(self, data: Batch | Observation) -> torch.Tensor:
        """
        Compute the qvalues for the given input data.
        - If the input is a `Batch`, the output should have an appropriate shape for the loss function
        - If the input is an `Observation`, the output shoud have shape (n_agents, n_actions)
        """

    @abstractmethod
    def compute_targets(self, batch: Batch) -> torch.Tensor:
        """Compute the target Qvalues for the given batch with Bellman's equation."""

    @abstractmethod
    def compute_loss(self, qvalues: torch.Tensor, qtargets: torch.Tensor, batch: Batch) -> torch.Tensor:
        """Computes and returns the loss"""

    @abstractmethod
    def process_batch(self, batch: Batch) -> Batch:
        """Process the batch sampled from the replay buffer.
        This method also applies every modfication to the batch that is necessary for the algorithm.
        e.g: batch.for_independent_learners(), ...
        """

    @property
    @abstractmethod
    def memory(self) -> ReplayMemory:
        """The attached replay memory"""
