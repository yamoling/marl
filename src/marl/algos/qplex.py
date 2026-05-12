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

    def get_mixing_kwargs(self, batch: Batch, all_qvalues: torch.Tensor, is_next: bool = False):
        kwargs = super().get_mixing_kwargs(batch, all_qvalues, is_next)
        if is_next:
            qplex_args = {"all_qvalues": all_qvalues, "available_actions": batch.next_available_actions}
            raise NotImplementedError("TODO: check how to implemebt the next_one_hot_acitons")
        else:
            qplex_args = {
                "all_qvalues": all_qvalues,
                "available_actions": batch.available_actions,
                "one_hot_actions": batch.one_hot_actions,
            }
        return kwargs | qplex_args
