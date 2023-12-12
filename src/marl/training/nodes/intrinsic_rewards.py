from marl.intrinsic_reward import IRModule
from marl.models import Batch
import torch
from .node import Node

class IR(Node[Batch]):
    def __init__(self, ir_module: IRModule, batch: Node[Batch]):
        super().__init__([batch])
        self.ir_module = ir_module
        self.batch = batch

    def _compute_value(self) -> Batch:
        batch = self.batch.value
        ir = self.ir_module.compute(batch)
        batch.rewards += ir
        return batch


    def to(self, device: torch.device):
        self.ir_module.to(device)
        super().to(device)