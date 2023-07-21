import torch
from marl.qlearning.mixers import Mixer
from marl.models import Batch
from .node import Node


class MixNode(Node[torch.Tensor]):
    def __init__(self, mixer: Mixer, qvalues: Node[torch.Tensor], batch: Node[Batch]):
        super().__init__()
        self.parent = qvalues
        self.mixer = mixer
        self.batch = batch

    @property
    def value(self) -> torch.Tensor:
        return self.mixer.forward(self.parent.value, self.batch.value.states_)
    
