from dataclasses import dataclass

import torch

from marl.models import Batch
from marl.nn import mixers

from .dqn import DQN


@dataclass
class QPlex(DQN[mixers.QPlex]):
    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.mixer, mixers.QPlex), "QPlex training requires a QPlex mixer"

    def get_mixing_kwargs(
        self,
        batch: Batch,
        all_qvalues: torch.Tensor,
        is_next: bool = False,
        actions: torch.Tensor | None = None,
    ):
        kwargs = super().get_mixing_kwargs(batch, all_qvalues, is_next, actions)
        if actions is None:
            raise ValueError("QPlex requires the selected actions when constructing mixer inputs")
        one_hot_actions = torch.nn.functional.one_hot(actions.long(), batch.n_actions).to(all_qvalues.dtype)
        if is_next:
            available_actions = batch.next_available_actions
        else:
            available_actions = batch.available_actions
        return kwargs | {
            "all_qvalues": all_qvalues,
            "available_actions": available_actions,
            "one_hot_actions": one_hot_actions,
        }

    def get_value_mixing_kwargs(self, all_qvalues: torch.Tensor) -> dict[str, torch.Tensor]:
        """Build QPlex inputs for a greedy value query. @ai-generated"""
        actions = all_qvalues.argmax(dim=-1)
        one_hot_actions = torch.nn.functional.one_hot(actions, self.qnetwork.n_actions).to(all_qvalues.dtype)
        return {"all_qvalues": all_qvalues, "one_hot_actions": one_hot_actions}
