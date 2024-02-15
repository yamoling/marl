import torch
from marl.models import Mixer, Batch
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
        mixed = self.mixer.forward(self.qvalues.value, self.batch.value.states)
        return mixed.squeeze(dim=-1)

    def randomize(self):
        self.mixer.randomize()
        return super().randomize()


class TargetQValueMixer(QValueMixer):
    def _compute_value(self) -> torch.Tensor:
        # with torch.no_grad():
        mixed = self.mixer.forward(self.qvalues.value, self.batch.value.states_)
        return mixed.squeeze(dim=-1)
