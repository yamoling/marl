from marl.intrinsic_reward import IRModule
from marl.models import Batch
from .node import Node

class IR(Node[Batch]):
    def __init__(self, ir_module: IRModule, batch: Node[Batch]):
        super().__init__([batch])
        self.ir_module = ir_module
        self.batch = batch

    @property
    def value(self) -> Batch:
        batch = self.batch.value
        ir = self.ir_module.compute(batch)
        batch.rewards += ir
        return batch
