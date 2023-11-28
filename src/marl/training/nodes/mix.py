import torch
from marl.qlearning.mixers import Mixer
from marl.models import Batch
from .node import Node


class QValueMixer(Node[torch.Tensor]):
    def __init__(self, mixer: Mixer, qvalues: Node[torch.Tensor], batch: Node[Batch]):
        super().__init__([qvalues, batch])
        self.qvalues = qvalues
        self.mixer = mixer
        self.batch = batch

    def to(self, device: torch.device):
        self.mixer.to(device)
        return super().to(device)

    def _compute_value(self) -> torch.Tensor:
        return self.mixer.forward(self.qvalues.value, self.batch.value.states)


class TargetQValueMixer(QValueMixer):
    def _compute_value(self) -> torch.Tensor:
        with torch.no_grad():
            return self.mixer.forward(self.qvalues.value, self.batch.value.states_)
