import torch
from typing import Literal
from rlenv import RLEnv
from marl.nn import model_bank, LinearNN, RecurrentNN
from marl.policy import Policy, EpsilonGreedy, ArgMax
from marl.utils import get_device
from marl.models import ReplayMemory, TransitionMemory, EpisodeMemory
from .qlearning import IDeepQLearning
from .dqn import DQN
from .vdn import VDN
from .rdqn import RDQN


class DeepQBuilder:
    def __init__(self, is_recurrent=False, env: RLEnv=None):
        self._env = env
        self._gamma = 0.99
        self._batch_size = 64
        self._tau = 1e-2
        self._qnetwork = None
        self._device = None
        self._optimizer = None
        self._train_policy = None
        self._test_policy = None
        self._memory = None

        self._is_recurrent = is_recurrent
        self._wrappers: list[Literal["vdn"]] = []
        if env is not None:
            self.qnetwork_default(env)
        

    def _fill_with_defaults(self):
        # Infer qnetwork if not provided
        if self._qnetwork is None:
            raise ValueError("No environment provided to infer qnetwork")
        if self._device is None:
            self._device = get_device('auto')
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self._qnetwork.parameters(), lr=1e-4)
        if self._train_policy is None:
            self._train_policy = EpsilonGreedy(0.1)
        if self._test_policy is None:
            self._test_policy = ArgMax()
        if self._memory is None:
            if self._is_recurrent:
                self._memory = TransitionMemory(50_000)
            else:
                self._memory = EpisodeMemory(50_000)

    def device(self, device):
        """Device to use for training (default: auto)"""
        self._device = device
        return self
        
    def qnetwork_default(self, env: RLEnv):
        if self._is_recurrent:
            self._qnetwork = model_bank.RNNQMix.from_env(env)
        else:
            self._qnetwork = model_bank.MLP.from_env(env)
        return self

    def qnetwork(self, qnetwork: LinearNN | RecurrentNN):
        if self._is_recurrent:
            assert isinstance(qnetwork, RecurrentNN)
        else:
            assert isinstance(qnetwork, LinearNN)
        self._qnetwork = qnetwork
        return self
    
    def memory(self, memory: ReplayMemory):
        self._memory = memory
        return self

    def gamma(self, gamma: float):
        """Discount factor (default: 0.99)"""
        self._gamma = gamma
        return self
    
    def tau(self, tau: float):
        """Target network soft update coefficient (default: 1e-2)"""
        self._tau = tau
        return self
    
    def batch_size(self, batch_size: int):
        """Batch size (default: 64)"""
        self._batch_size = batch_size
        return self

    def optimizer(self, optimizer: Literal["adam", "rmsprop"], lr=1e-4):
        assert self._qnetwork is not None, "Must provide qnetwork before optimizer"
        match optimizer:
            case "adam": self._optimizer = torch.optim.Adam(self._qnetwork.parameters(), lr=lr)
            case "rmsprop": self._optimizer = torch.optim.RMSprop(self._qnetwork.parameters(), lr=lr)
            case other: raise ValueError(f"Optimizer {other} not supported")
        return self
    
    def train_policy(self, policy: Policy):
        """Policy to use during training (default: EpsilonGreedy(0.1))"""
        self._train_policy = policy
        return self
    
    def test_policy(self, policy: Policy):
        """Policy to use during testing (default: ArgMax)"""
        self._test_policy = policy
        return self
    
    def vdn(self):
        self._wrappers.append("vdn")
        return self

    def build(self) -> IDeepQLearning:
        self._fill_with_defaults()
        if self._is_recurrent:
            algo = RDQN(
                qnetwork=self._qnetwork,
                gamma=self._gamma,
                tau=self._tau,
                memory=self._memory,
                batch_size=self._batch_size,
                device=self._device,
                optimizer=self._optimizer,
                test_policy=self._test_policy,
                train_policy=self._train_policy
            )
        else:
            algo = DQN(
                qnetwork=self._qnetwork,
                gamma=self._gamma,
                tau=self._tau,
                memory=self._memory,
                batch_size=self._batch_size,
                device=self._device,
                optimizer=self._optimizer,
                test_policy=self._test_policy,
                train_policy=self._train_policy
            )
        for wrapper in self._wrappers:
            match wrapper:
                case "vdn": algo = VDN(algo)
                case other: raise ValueError(f"Wrapper {other} not supported")
        return algo
