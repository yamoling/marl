from abc import abstractmethod
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from marlenv import DiscreteMARLEnv

from marl.env import EnvConfig

from .nn import NN

if TYPE_CHECKING:
    from marl.models import Batch


@dataclass
class Mixer(NN):
    output_shape: tuple[int, ...] = field(init=False)
    _: KW_ONLY
    n_objectives: int = 1

    def __post_init__(self):
        super().__post_init__()
        self.output_shape = (self.n_objectives,)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def agent_dim(self):
        if self.n_objectives == 1:
            return -1
        return -2

    @abstractmethod
    def forward(self, qvalues: torch.Tensor, states: torch.Tensor, states_extras: torch.Tensor, /, **kwargs) -> torch.Tensor:
        """
        Mix the utiliy values of the agents.

        To englobe every possible mixer, the signature of the forward method is quite complex.
        - qvalues: the Q-values of the action take by each agent. (batch, n_agents)
        - states: the state of the environment. (batch, state_size)
        """

    def mixing_kwargs(
        self,
        all_qvalues: torch.Tensor,
        actions: torch.Tensor,
        batch: "Batch | None" = None,
        *,
        is_next: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Additional keyword arguments that this mixer requires in `forward`, besides the Q-values and the states.

        Args:
            - all_qvalues: Q-values of every action, with shape (*dims, n_agents, n_actions[, n_objectives]).
            - actions: the action of each agent whose Q-value is mixed, with shape (*dims, n_agents).
            - batch: the batch the Q-values come from, if any (e.g. None when evaluating a single state).
            - is_next: whether the Q-values relate to the next states of the batch rather than the current ones.
        """
        return {}

    def forward_batch(
        self,
        qvalues: torch.Tensor,
        batch: "Batch",
        all_qvalues: torch.Tensor,
        actions: torch.Tensor,
        *,
        is_next: bool = False,
    ) -> torch.Tensor:
        """Mix the Q-values of the given (current or next) time steps of a batch, including the inputs of `mixing_kwargs`."""
        if is_next:
            states, states_extras = batch.next_states, batch.next_states_extras
        else:
            states, states_extras = batch.states, batch.states_extras
        return self.forward(qvalues, states, states_extras, **self.mixing_kwargs(all_qvalues, actions, batch, is_next=is_next))

    def save(self, directory: Path):
        """Save the mixer to a directory."""
        filename = f"{directory}/mixer.weights"
        state_dict = self.state_dict()
        if len(state_dict) == 0:
            return
        torch.save(state_dict, filename)

    def weights_filename(self, directory: Path):
        return directory / "mixer.weights"

    @classmethod
    def from_env(cls, env: DiscreteMARLEnv | EnvConfig[DiscreteMARLEnv], **kwargs):
        """Create a mixer from an environment."""
        return cls(n_objectives=env.n_objectives, **kwargs)


@dataclass
class StateMixer(Mixer):
    n_agents: int
    state_size: int
    state_extras_size: int

    @classmethod
    def from_env(cls, env: DiscreteMARLEnv | EnvConfig[DiscreteMARLEnv], **kwargs):
        return super().from_env(env, n_agents=env.n_agents, state_size=env.state_size, state_extras_size=env.state_extras_size, **kwargs)
