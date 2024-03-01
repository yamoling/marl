import torch
from marl.models import Mixer, Batch
from .node import Node


class QValueMixer(Node[torch.Tensor]):
    def __init__(self, mixer: Mixer, selected_qvalues: Node[torch.Tensor], all_qvalues: Node[torch.Tensor], batch: Node[Batch]):
        super().__init__([selected_qvalues, batch])
        self.selected_qvalues = selected_qvalues
        self.all_qvalues = all_qvalues
        self.mixer = mixer
        self.batch = batch

    def to(self, device: torch.device):
        self.mixer.to(device)
        return super().to(device)

    def _compute_value(self) -> torch.Tensor:
        mixed = self.mixer.forward(
            self.selected_qvalues.value,
            self.batch.value.states,
            self.batch.value.one_hot_actions,
            self.all_qvalues.value,
        )
        return mixed.squeeze(dim=-1)

    def randomize(self):
        self.mixer.randomize()
        return super().randomize()


class TargetQValueMixer(QValueMixer):
    def _compute_value(self) -> torch.Tensor:
        mixed = self.mixer.forward(
            self.selected_qvalues.value,
            self.batch.value.states_,
            self.batch.value.one_hot_actions,
            self.all_qvalues.value,
        )
        return mixed.squeeze(dim=-1)
