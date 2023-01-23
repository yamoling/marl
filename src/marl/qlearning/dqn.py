from dataclasses import dataclass
from copy import deepcopy
import torch
from rlenv import RLEnv, Transition, Observation
from marl import RLAlgorithm, nn
from marl.models import TransitionMemory, Batch
from marl.utils import get_device
from marl.policy import Policy, EpsilonGreedy
from marl.utils import defaults_to

from .qlearning import QLearning

@dataclass
class DQN(QLearning):
    """
    Independent Deep Q-Network agent with shared QNetwork.
    If agents require different behaviours, an agentID should be included in the 
    observation 'extras'.
    """
    gamma: float
    tau: float
    batch_size: int
    lr: float
    qnetwork: nn.LinearNN
    optimizer: torch.optim.Optimizer
    policy: Policy
    memory: TransitionMemory
    device: torch.device

    def __init__(
        self, 
        env: RLEnv, 
        gamma=0.99,
        tau=1e-2,
        batch_size=64,
        lr=1e-4,
        qnetwork: nn.LinearNN=None,
        optimizer: torch.optim.Optimizer=None,
        train_policy: Policy=None,
        test_policy: Policy=None,
        memory: TransitionMemory=None,
        device: torch.device=None
    ):
        self.qnetwork = defaults_to(qnetwork, nn.model_bank.MLP.from_env(env))
        self.qtarget = deepcopy(self.qnetwork)
        self.gamma = gamma
        self.tau = tau
        """Soft update tau value"""
        self.loss_function = torch.nn.MSELoss()
        super().__init__(
            env=env, 
            memory=defaults_to(memory, TransitionMemory(50_000)),
            batch_size=batch_size, 
            optimizer=defaults_to(optimizer, torch.optim.Adam(self.qnetwork.parameters(), lr=lr)), 
            device=device,
            train_policy=defaults_to(train_policy, EpsilonGreedy(env.n_agents, 0.1)),
            test_policy=defaults_to(test_policy, EpsilonGreedy(env.n_agents, 0.01))
        )
    
    def after_step(self, _step_num: int, transition: Transition):
        self.memory.add(transition)
        self.update()
    
    def compute_qvalues(self, data: Batch|Observation) -> torch.Tensor:
        match data:
            case Batch() as batch:
                qvalues = self.qnetwork.forward(batch.obs, batch.extras)
                qvalues = qvalues.gather(index=batch.actions, dim=-1)
                return qvalues.squeeze(dim=-1)
            case Observation() as obs:
                obs_data = torch.from_numpy(obs.data).to(self.device, non_blocking=True)
                obs_extras = torch.from_numpy(obs.extras).to(self.device, non_blocking=True)
                return self.qnetwork.forward(obs_data, obs_extras)
            case _: raise ValueError("Invalid input data type for 'compute_qvalues'")

    def compute_targets(self, batch: Batch) -> torch.Tensor:
        next_qvalues = self.qtarget.forward(batch.obs_, batch.extras_)
        next_qvalues[batch.available_actions_ == 0.0] = -torch.inf
        next_qvalues = torch.max(next_qvalues, dim=-1)[0]
        target_qvalues = batch.rewards + self.gamma * next_qvalues * (1 - batch.dones)
        return target_qvalues

    def compute_loss(self, qvalues: torch.Tensor, qtargets: torch.Tensor, _batch: Batch) -> torch.Tensor:
        return self.loss_function(qvalues, qtargets)

    def _sample(self) -> Batch:
        batch = self.memory.sample(self.batch_size)
        batch.for_individual_learners()
        return batch

    def update(self):
        super().update()
        self._target_soft_update()

    def _target_soft_update(self):
        for param, target_param in zip(self.qnetwork.parameters(), self.qtarget.parameters()):
            new_value = (1-self.tau) * target_param.data + self.tau * param.data
            target_param.data.copy_(new_value, non_blocking=True)

    def save(self, to_path: str):
        pass

