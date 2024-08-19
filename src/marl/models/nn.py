from typing import Optional
from dataclasses import dataclass
from typing import Literal
from rlenv import Observation
from abc import ABC, abstractmethod
import torch
from serde import serde

from rlenv.models import RLEnv


def randomize(init_fn, nn: torch.nn.Module):
    for param in nn.parameters():
        if len(param.data.shape) < 2:
            init_fn(param.data.view(1, -1))
        else:
            init_fn(param.data)


@serde
@dataclass
class NN(torch.nn.Module, ABC):
    """Parent class of all neural networks"""

    input_shape: tuple[int, ...]
    extras_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    name: str

    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], output_shape: tuple[int, ...]):
        torch.nn.Module.__init__(self)
        self.input_shape = input_shape
        self.extras_shape = extras_shape
        self.output_shape = output_shape
        self.name = self.__class__.__name__
        self.device = torch.device("cpu")

    @abstractmethod
    def forward(self, *args) -> torch.Tensor:
        """Forward pass"""

    def randomize(self, method: Literal["xavier", "orthogonal"] = "xavier"):
        match method:
            case "xavier":
                randomize(torch.nn.init.xavier_uniform_, self)
            case "orthogonal":
                randomize(torch.nn.init.orthogonal_, self)
            case _:
                raise ValueError(f"Unknown initialization method: {method}. Choose between 'xavier' and 'orthogonal'")

    @classmethod
    def from_env(cls, env: RLEnv):
        """Construct a NN from environment specifications"""
        return cls(input_shape=env.observation_shape, extras_shape=env.extra_feature_shape, output_shape=(env.n_actions,))

    def __hash__(self) -> int:
        # Required for deserialization (in torch.nn.module)
        return hash(self.__class__.__name__)

    def to(self, device: torch.device | Literal["cpu", "auto"], dtype: Optional[torch.dtype] = None, non_blocking=True):
        if isinstance(device, str):
            from marl.utils import get_device

            device = get_device(device)
        self.device = device
        return super().to(device, dtype, non_blocking)


class RecurrentNN(NN):
    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], output_shape: tuple[int, ...]):
        super().__init__(input_shape, extras_shape, output_shape)
        self.hidden_states: Optional[torch.Tensor] = None
        self.saved_hidden_states = None

    def reset_hidden_states(self):
        """Reset the hidden states"""
        self.hidden_states = None

    def train(self, mode: bool = True):
        if not mode:
            # Set test mode: save training hidden states
            self.saved_hidden_states = self.hidden_states
            self.reset_hidden_states()
        else:
            # Set train mode
            if not self.training:
                # if not already in train mode, restore hidden states
                self.hidden_states = self.saved_hidden_states
        return super().train(mode)


class QNetwork(NN):
    """
    Takes as input observations of the environment and outputs Q-values for each action.
    """

    def to_tensor(self, obs: Observation) -> tuple[torch.Tensor, torch.Tensor]:
        extras = torch.from_numpy(obs.extras).unsqueeze(0).to(self.device)
        obs_tensor = torch.from_numpy(obs.data).unsqueeze(0).to(self.device)
        return obs_tensor, extras

    def qvalues(self, obs: Observation) -> torch.Tensor:
        """
        Compute the Q-values (one per agent, per action and per objective).

        The resulting shape is (n_agents, n_actions, n_objectives)
        """
        obs_tensor, extra_tensor = self.to_tensor(obs)
        objective_qvalues = self.forward(obs_tensor, extra_tensor)
        objective_qvalues = objective_qvalues.squeeze(0)
        return objective_qvalues

    def value(self, obs: Observation) -> torch.Tensor:
        """Compute the value function (maximum of the q-values)."""
        objective_qvalues = self.qvalues(obs)
        qvalues = torch.sum(objective_qvalues, dim=-1)
        agent_values = qvalues.max(dim=-1).values
        return agent_values.mean(dim=-1)

    @abstractmethod
    def forward(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        """
        Compute the Q-values.

        This function should output qvalues of shape (batch_size, n_actions, n_objectives).
        """

    def batch_forward(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        """Compute the Q-values for a batch of observations during training"""
        return self.forward(obs, extras)

    def set_testing(self, test_mode: bool = True):
        """Set the network in testing mode"""
        self.test_mode = test_mode

    @classmethod
    def from_env(cls, env: RLEnv):
        return cls(
            input_shape=env.observation_shape,
            extras_shape=env.extra_feature_shape,
            output_shape=(env.n_actions, env.reward_size),
        )


class RecurrentQNetwork(QNetwork, RecurrentNN):
    def value(self, obs: Observation) -> torch.Tensor:
        """Compute the value function. Does not update the hidden states."""
        hidden_states = self.hidden_states
        objective_qvalues = self.qvalues(obs)
        qvalues = torch.sum(objective_qvalues, dim=-1)
        agent_values = torch.max(qvalues, dim=-1).values
        self.hidden_states = hidden_states
        return agent_values.mean(dim=-1)

    def batch_forward(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        """
        Compute the Q-values for a batch of observations (multiple episodes) during training.

        In this case, the RNN considers hidden states=None.
        """
        self.test_mode = False
        saved_hidden_states = self.hidden_states
        self.reset_hidden_states()
        qvalues = self.forward(obs, extras)
        self.hidden_states = saved_hidden_states
        return qvalues


class ActorCriticNN(NN, ABC):
    """Actor critic neural network"""

    @property
    @abstractmethod
    def policy_parameters(self) -> list[torch.nn.Parameter]:
        pass

    @property
    @abstractmethod
    def value_parameters(self) -> list[torch.nn.Parameter]:
        pass

    @abstractmethod
    def policy(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns the logits of the policy distribution"""

    @abstractmethod
    def value(self, obs: torch.Tensor, extras: torch.Tensor, *args) -> torch.Tensor:
        """Returns the value function of an observation"""

    def forward(self, obs: torch.Tensor, extras: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the logits of the policy distribution and the value function given an observation"""
        return self.policy(obs), self.value(obs, extras)


@dataclass(eq=False)
class Mixer(NN):
    n_agents: int

    def __init__(self, n_agents: int):
        super().__init__((n_agents,), (0,), (1,))
        self.n_agents = n_agents

    @abstractmethod
    def forward(
        self,
        qvalues: torch.Tensor,
        states: torch.Tensor,
        one_hot_actions: torch.Tensor,
        all_qvalues: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mix the utiliy values of the agents.

        To englobe every possible mixer, the signature of the forward method is quite complex.
        - qvalues: the Q-values of the action take by each agent. (batch, n_agents)
        - states: the state of the environment. (batch, state_size)
        - one_hot_actions: the action taken by each agent. (batch, n_agents, 1)
        - all_qvalues: all Q-values of the agents. (batch, n_agents, n_actions)
        """

    def save(self, to_directory: str):
        """Save the mixer to a directory."""
        filename = f"{to_directory}/mixer.weights"
        torch.save(self.state_dict(), filename)

    def load(self, from_directory: str):
        """Load the mixer from a directory."""
        filename = f"{from_directory}/mixer.weights"
        self.load_state_dict(torch.load(filename))


class MAIC(ABC):
    @abstractmethod
    def get_values_and_comms(
        self, obs: torch.Tensor, extras: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the Q-values and return Q-values and Computed messages"""


class MAICNN(NN):
    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], output_shape: tuple[int, ...]):
        super().__init__(input_shape, extras_shape, output_shape)
        self.hidden_states: Optional[torch.Tensor] = None
        self.saved_hidden_states = None

    def reset_hidden_states(self):
        """Reset the hidden states"""
        self.hidden_states = None

    def train(self, mode: bool = True):
        if not mode:
            # Set test mode: save training hidden states
            self.saved_hidden_states = self.hidden_states
            self.reset_hidden_states()
        else:
            # Set train mode
            if not self.training:
                # if not already in train mode, restore hidden states
                self.hidden_states = self.saved_hidden_states
        return super().train(mode)

    def to_tensor(self, obs: Observation) -> tuple[torch.Tensor, torch.Tensor]:
        extras = torch.from_numpy(obs.extras).unsqueeze(0).to(self.device)
        obs_tensor = torch.from_numpy(obs.data).unsqueeze(0).to(self.device)
        return obs_tensor, extras

    def qvalues(self, obs: Observation):
        """Compute the Q-values"""
        obs_tensor, extra_tensor = self.to_tensor(obs)
        objective_qvalues, _ = self.forward(obs_tensor, extra_tensor)
        objective_qvalues = objective_qvalues.squeeze(0)
        return objective_qvalues

    def value(self, obs: Observation) -> torch.Tensor:
        """Compute the value function. Does not update the hidden states."""
        hidden_states = self.hidden_states
        objective_qvalues = self.qvalues(obs)
        qvalues = torch.sum(objective_qvalues, dim=-1)
        agent_values = torch.max(qvalues, dim=-1).values
        self.hidden_states = hidden_states
        return agent_values.mean(dim=-1)

    @abstractmethod
    def forward(self, obs: torch.Tensor, extras: torch.Tensor) -> ...:
        """Compute the Q-values"""

    def set_testing(self, test_mode: bool = True):
        """Set the network in testing mode"""
        self.test_mode = test_mode

    def batch_forward(self, obs: torch.Tensor, extras: torch.Tensor):
        """
        Compute the Q-values for a batch of observations (multiple episodes) during training.

        In this case, the RNN considers hidden states=None.
        """
        self.test_mode = False
        saved_hidden_states = self.hidden_states
        self.reset_hidden_states()
        logs = []
        losses = []
        q_values, returns_ = self.forward(obs, extras)
        if "logs" in returns_:
            logs.append(returns_["logs"])
            del returns_["logs"]
        losses.append(returns_)
        self.hidden_states = saved_hidden_states
        return q_values, logs, losses
        # self.test_mode = False
        # bs = obs.shape[1]
        # self.reset_hidden_states(bs)
        # qvalues = []
        # logs = []
        # losses = []
        # for t in range(len(obs)):  # case of Episode Batch
        #     current_q_values, returns_ = self.forward(obs[t], extras[t])
        #     qvalues.append(current_q_values)
        #     if "logs" in returns_:
        #         logs.append(returns_["logs"])
        #         del returns_["logs"]
        #     losses.append(returns_)

        # qvalues = torch.stack(qvalues, dim=0)

        # return qvalues, logs, losses
