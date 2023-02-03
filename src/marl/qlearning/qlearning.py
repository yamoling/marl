from abc import abstractmethod
import torch
from marl import RLAlgorithm
from marl.policy import Policy
from marl.models import ReplayMemory, Batch
from marl.utils import defaults_to, get_device
from rlenv import RLEnv, Observation


class QLearning(RLAlgorithm):
    def __init__(self, env: RLEnv, test_env: RLEnv, log_path: str = None, train_policy: Policy=None, test_policy:Policy=None, gamma=0.99) -> None:
        super().__init__(env, test_env, log_path)
        self.train_policy = train_policy
        self.test_policy = test_policy
        self.policy = train_policy
        self.gamma = gamma

    def before_tests(self):
        self.policy = self.test_policy
        return super().before_tests()

    def after_tests(self, time_step: int, episodes):
        self.policy = self.train_policy
        return super().after_tests(time_step, episodes)

    def choose_action(self, obs: Observation) -> list[int]:
        with torch.no_grad():
            qvalues = self.compute_qvalues(obs)
            qvalues = qvalues.cpu().numpy()
        return self.policy.get_action(qvalues, obs.available_actions)

    @abstractmethod
    def compute_qvalues(self, data: Observation) -> torch.Tensor:
        """Compute the qvalues for the given input data."""
    
    


class DeepQLearning(QLearning):
    def __init__(
        self, 
        env: RLEnv, 
        test_env: RLEnv,
        train_policy: Policy, 
        test_policy: Policy,
        memory: ReplayMemory, 
        batch_size: int, 
        optimizer: torch.optim.Optimizer, 
        device: torch.device=None,
        log_path: str=None
    ) -> None:
        super().__init__(env, test_env=test_env, log_path=log_path)
        self.memory = memory
        self.batch_size = batch_size
        self.device = defaults_to(device, get_device())
        self.optimizer = optimizer
        self.train_policy = train_policy
        self.test_policy = test_policy
        self.policy = train_policy
        """The current policy (either train or test)"""

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
    def _sample(self) -> Batch:
        """Sample from the replay buffer.
        This method also applies every modfication to the batch that is necessary for the algorithm.
        e.g: batch.for_rnn(), batch.for_independent_learners(), ...
        """
        pass

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self._sample().to(self.device)
        # Compute qvalues and qtargets (delegated to child classes)
        qvalues = self.compute_qvalues(batch)
        with torch.no_grad():
            qtargets = self.compute_targets(batch)
        assert qvalues.shape == qtargets.shape, f"Predicted qvalues ({qvalues.shape}) and target qvalues ({qtargets.shape}) do not have the same shape !"
        # Compute the loss and apply gradient descent
        loss = self.compute_loss(qvalues, qtargets, batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.train_policy.update()
