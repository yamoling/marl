from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch

@dataclass
class Batch(ABC):
    """
    Lazy loaded batch for training.
    Every field is set to None by default. When the field is accessed, the specific attribute is 
    loaded with the `_get_<attribute>()` method that child classes must implement.
    (exception for importance sampling weights that are set by the memory)
    """
    _obs: torch.Tensor
    _obs_: torch.Tensor
    _extras: torch.Tensor
    _extras_: torch.Tensor
    _actions: torch.LongTensor
    _rewards: torch.Tensor
    _dones: torch.Tensor
    _available_actions: torch.LongTensor
    _available_actions_: torch.LongTensor
    _states: torch.Tensor
    _states_: torch.Tensor
    _action_probs: torch.Tensor
    _sampled_indices: list[int]
    importance_sampling_weights: torch.Tensor | None

    def __init__(self, size: int, n_agents: int, sample_indices: list[int]) -> None:
        super().__init__()
        self.size = size
        self.n_agents = n_agents
        self.sample_indices = sample_indices
        self.device = torch.device("cpu")
        self._obs = None
        self._obs_ = None
        self._extras = None
        self._extras_ = None
        self._actions = None
        self._rewards = None
        self._dones = None
        self._available_actions = None
        self._available_actions_ = None
        self._states = None
        self._states_ = None
        self._action_probs = None
        self.importance_sampling_weights = None

    def for_individual_learners(self) -> "Batch":
        """Reshape rewards, dones such that each agent has its own (identical) signal."""
        self._rewards = self.rewards.repeat_interleave(self.n_agents).view(*self.rewards.shape, self.n_agents)
        self._dones = self.dones.repeat_interleave(self.n_agents).view(*self.dones.shape, self.n_agents)
        return self

    def __len__(self) -> int:
        return self.size

    @property
    def obs(self) -> torch.Tensor:
        """Observations"""
        if self._obs is None:
            self._obs = self._get_obs().to(self.device, non_blocking=True)
        return self._obs

    @property
    def obs_(self) -> torch.Tensor:
        """Next observations"""
        if self._obs_ is None:
            self._obs_ = self._get_obs_().to(self.device, non_blocking=True)
        return self._obs_

    @property
    def extras(self) -> torch.Tensor:
        """Extra information"""
        if self._extras is None:
            self._extras = self._get_extras().to(self.device, non_blocking=True)
        return self._extras

    @property
    def extras_(self) -> torch.Tensor:
        """Nest extra information"""
        if self._extras_ is None:
            self._extras_ = self._get_extras_().to(self.device, non_blocking=True)
        return self._extras_

    @property
    def actions(self) -> torch.LongTensor:
        """Actions"""
        if self._actions is None:
            self._actions = self._get_actions().to(self.device, non_blocking=True)
        return self._actions

    @property
    def rewards(self) -> torch.Tensor:
        """Rewards"""
        if self._rewards is None:
            self._rewards = self._get_rewards().to(self.device, non_blocking=True)
        return self._rewards

    @property
    def dones(self) -> torch.Tensor:
        """Dones"""
        if self._dones is None:
            self._dones = self._get_dones().to(self.device, non_blocking=True)
        return self._dones

    @property
    def available_actions_(self) -> torch.LongTensor:
        """Next available actions"""
        if self._available_actions_ is None:
            self._available_actions_ = self._get_available_actions_().to(self.device, non_blocking=True)
        return self._available_actions_

    @property
    def available_actions(self) -> torch.LongTensor:
        """Available actions"""
        if self._available_actions is None:
            self._available_actions = self._get_available_actions().to(self.device, non_blocking=True)
        return self._available_actions

    @property
    def states(self) -> torch.Tensor:
        """Environment states"""
        if self._states is None:
            self._states = self._get_states().to(self.device, non_blocking=True)
        return self._states

    @property
    def states_(self) -> torch.Tensor:
        """Next environment states"""
        if self._states_ is None:
            self._states_ = self._get_states_().to(self.device, non_blocking=True)
        return self._states_

    @property
    def action_probs(self) -> torch.Tensor:
        """Probabilities of the taken action"""
        if self._action_probs is None:
            self._action_probs = self._get_action_probs().to(self.device, non_blocking=True)
        return self._action_probs
    
    def to(self, device: torch.device) -> "Batch":
        """Send the tensors to the given device"""
        self.device = device
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                value = value.to(device, non_blocking=True)
                setattr(self, key, value)
        return self
    
    @abstractmethod
    def _get_obs(self) -> torch.Tensor:
        pass

    @abstractmethod
    def _get_obs_(self) -> torch.Tensor:
        pass

    @abstractmethod
    def _get_extras(self) -> torch.Tensor:
        pass

    @abstractmethod
    def _get_extras_(self) -> torch.Tensor:
        pass

    @abstractmethod
    def _get_actions(self) -> torch.LongTensor:
        pass

    @abstractmethod
    def _get_rewards(self) -> torch.Tensor:
        pass

    @abstractmethod
    def _get_dones(self) -> torch.Tensor:
        pass

    @abstractmethod
    def _get_available_actions_(self) -> torch.LongTensor:
        pass

    @abstractmethod
    def _get_available_actions(self) -> torch.LongTensor:
        pass

    @abstractmethod
    def _get_states(self) -> torch.Tensor:
        pass

    @abstractmethod
    def _get_states_(self) -> torch.Tensor:
        pass

    @abstractmethod
    def _get_action_probs(self) -> torch.Tensor:
        pass

