from typing import Set

import torch

from rlenv import Transition
from marl.models import Batch, TransitionMemory
from marl.nn import NN
from .node import Node, ValueNode
from .standard import LossNode


class DQNTrainer:
    """Qlearning trainer based on a computation graph"""

    def __init__(
        self,
        memory: TransitionMemory,
        optimizer: torch.optim.Optimizer,
        batch_node: ValueNode[Batch],
        loss_node: LossNode,
        networks: list[NN],
        targets: list[NN],
        batch_size: int=64,
        train_interval: int=5,
        tau: float=0.01,
    ) -> None:
        self._memory = memory
        self._batch = batch_node
        self._loss = loss_node
        self._optimizer = optimizer
        self._train_interval = train_interval
        self._models = networks
        self._targets = targets
        self._tau = tau
        self._batch_size = batch_size
        self._device = networks[0].device

    def update(self, transition: Transition, step_num: int):
        self._memory.add(transition)
        if len(self._memory) < self._batch_size or step_num % self._train_interval != 0:
            return
        self._batch.value = self._memory.sample(self._batch_size).to(self._device)
        loss = self._loss.value
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        for model, target in zip(self._models, self._targets):
            self._soft_update(model, target)
        
    def _soft_update(self, model: NN, target: NN):
        for param, target_param in zip(model.parameters(), target.parameters()):
            new_value = (1 - self._tau) * target_param.data + self._tau * param.data
            target_param.data.copy_(new_value, non_blocking=True)



    def show(self):
        """Display the computation graph"""
        import networkx as nx
        import matplotlib.pyplot as plt

        edges = []
        to_visit: Set[Node] = set([self._batch])
        visited = set()
        while len(to_visit) > 0:
            current = to_visit.pop()
            for child in current.children:
                edges.append((current.name, child.name))
                if child not in visited:
                    to_visit.add(child)
            visited.add(current)
        graph = nx.DiGraph()
        graph.add_edges_from(edges)
        pos = nx.spectral_layout(graph)
        nx.draw_networkx(graph, pos)
        plt.show()
