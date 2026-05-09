import math
from abc import abstractmethod
from dataclasses import KW_ONLY, dataclass, field
from typing import Literal

import torch
from marlenv import DiscreteMARLEnv
from torch import Tensor

from marl.env import EnvConfig
from marl.models.nn import NN, QNetwork

from ..generic import CNN, MLP


@dataclass
class MAVENTail(torch.nn.Module):
    """
    Tail of the MAVEN agent-wise network. The paper only presents the "hyper-network" approach
    but the official code proposes two kinds of networks. In either case, the hypernetwork:
    - BMM: generates weights for a BMM with the previous layer outputs
    - Multipyl: generates weights and performs an element-wise multiplication
    """

    noise_size: int
    n_agents: int
    agent_output_size: int
    n_actions: int

    def __post_init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, noise: Tensor, agent_output: Tensor) -> Tensor: ...

    def __hash__(self):
        return id(self)


@dataclass
class MAVENHyperBMM(MAVENTail):
    """
    This tail network is the approach presented in the MAVEN paper, i.e. a hyper-network that generates the weights to compute the q-values directly from the noise and agent ids.
    """

    def __post_init__(self):
        super().__post_init__()
        self.hyper_network = torch.nn.Linear(
            self.noise_size + self.n_agents,
            self.agent_output_size * self.n_actions,
        )

    def forward(self, noise: Tensor, agent_output: Tensor) -> Tensor:
        """
        The hyper-network takes as input the noise and the agent id and produces the weight matrix that will be multiplied with the previous layer outputs.
        The final output is of shape (batch_size, n_agents, n_actions), i.e. a q-value per agent and per action.
        """
        *dims, n_agents, noise_size = noise.shape
        batch_size = math.prod(dims)
        # Build the hyper-network inputs: [noise, agent_id]
        agent_ids = torch.eye(self.n_agents, device=noise.device).unsqueeze(0).repeat(batch_size, 1, 1)
        noise = noise.reshape(batch_size, n_agents, noise_size)
        inputs = torch.cat([noise, agent_ids], dim=-1)
        # The hyper-network takes as input the [noise, agent_id] and outputs HIDDEN_DIM * n_actions weights.
        weights = self.hyper_network.forward(inputs)
        # Reshape to match the batch matrix multiplication requirements
        # Agent_output: (batch_size, n_agents, agent_output) -> (batch_size * n_agents, 1, agent_output)
        # Weights     : (batch_size, n_agents, agent_output * n_actions) -> (batch_size * n_agents, agent_output, n_actions)
        weights = weights.view(batch_size * self.n_agents, self.agent_output_size, self.n_actions)
        agent_output = agent_output.view(batch_size * self.n_agents, 1, self.agent_output_size)
        res = torch.bmm(agent_output, weights)
        # Return in the original shape
        return res.view(*dims, self.n_agents, self.n_actions)

    def __hash__(self):
        return id(self)


@dataclass
class MAVENHyperMult(MAVENTail):
    def __post_init__(self):
        super().__post_init__()
        self.linear = torch.nn.Linear(self.agent_output_size, self.n_actions)
        self.mult_weights_nn = torch.nn.Sequential(
            torch.nn.Linear(self.noise_size + self.n_agents, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, self.n_actions),
        )

    def forward(self, noise: Tensor, agent_output: Tensor) -> Tensor:
        *dims, n_agents, noise_size = noise.shape
        batch_size = math.prod(dims)
        agent_ids = torch.eye(self.n_agents, device=noise.device).unsqueeze(0).repeat(batch_size, 1, 1)
        noise = noise.reshape(batch_size, n_agents, noise_size)
        qs = self.linear.forward(agent_output)
        inputs = torch.cat([noise, agent_ids], dim=-1)
        weights = self.mult_weights_nn.forward(inputs)
        return qs * weights

    def __hash__(self):
        return id(self)


@dataclass
class MAVENQnetwork(QNetwork):
    """
    MAVEN Q-Networks are composed of a standard head like any other DQN variant, but have a tail that
    generates weights for the final layer from the noise input and agent ids. This allows the noise to
    directly influence the q-values and thus the policy, which promotes exploration.
    """

    noise_size: int
    n_agents: int
    _: KW_ONLY
    head: NN = field(init=False)
    tail_type: Literal["bmm", "mul"] = "bmm"
    agent_output_size: int = 128

    def __post_init__(self):
        super().__post_init__()
        match self.obs_shape:
            case (_, _, _):
                self.head = CNN((self.agent_output_size,), self.obs_shape, self.actual_extras_size, output_activation="relu")
            case (_,):
                self.head = MLP((self.agent_output_size,), self.obs_size, self.actual_extras_size, output_activation="relu")
            case _:
                raise NotImplementedError(f"Observation shape {self.obs_shape} not supported for MAVEN.")
        match self.tail_type:
            case "bmm":
                self.tail = MAVENHyperBMM(self.noise_size, self.n_agents, self.agent_output_size, self.n_actions)
            case "mul":
                self.tail = MAVENHyperMult(self.noise_size, self.n_agents, self.agent_output_size, self.n_actions)
            case other:
                raise ValueError(f"Unknown hyper network type {other}")

    @property
    def actual_extras_size(self) -> int:
        return self.extras_size - self.noise_size

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs) -> torch.Tensor:
        match len(extras.shape):
            case 3:
                noise = extras[:, :, -self.noise_size :]
                extras = extras[:, :, : -self.noise_size]
            case 4:
                noise = extras[:, :, :, -self.noise_size :]
                extras = extras[:, :, :, : -self.noise_size]
            case _:
                raise NotImplementedError()
        x = self.head.forward(obs, extras, **kwargs)
        return self.tail.forward(noise, x)

    @classmethod
    def from_env(
        cls,
        env: DiscreteMARLEnv | EnvConfig[DiscreteMARLEnv],
        agent_output_size: int = 128,
        tail_type: Literal["bmm", "mul"] = "bmm",
        **kwargs,
    ):
        if not isinstance(env, EnvConfig):
            env = EnvConfig.from_any(env)
        return MAVENQnetwork(
            env.n_actions,
            env.observation_shape,
            env.extras_shape,
            env.noise_size,
            env.n_agents,
            agent_output_size=agent_output_size,
            tail_type=tail_type,
            **kwargs,
        )

    def __hash__(self):
        return id(self)
