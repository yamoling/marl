from copy import deepcopy
from dataclasses import dataclass
from serde import serde
from typing import Literal, Optional
from typing_extensions import Self

import os
import torch
from rlenv import Episode, Transition

from marl.intrinsic_reward import IRModule
from marl.models import Batch, EpisodeMemory, ReplayMemory, TransitionMemory
from marl.nn import LinearNN
from marl.qlearning.mixers import Mixer
from marl.training import nodes
from marl.utils import defaults_to, get_device

@serde
@dataclass
class DQNTrainer:
    qnetwork: LinearNN
    memory: Optional[ReplayMemory] = None
    gamma: float = 0.99
    batch_size: int = 64
    train_interval: int = 5
    lr: float = 1e-4
    optimizer_str: Literal["adam", "rmsprop"] = "adam"
    tau: float = 0.01
    double_qlearning: bool = False
    mixer: Optional[Mixer] = None
    device_str: Literal["auto", "cpu", "cuda"] = "auto"
    update_frequency: Literal["step", "episode"] = "step"
    ir_module: Optional[IRModule] = None

    def __post_init__(self):
        """Called after the dataclass init. All the non-parameter attributes are initialized here"""
        self.device = get_device(self.device_str)
        self.qnetwork = self.qnetwork.to(self.device)
        self.qtarget = deepcopy(self.qnetwork).to(self.device)
        self.parameters = list(self.qnetwork.parameters())
        self.target_parameters = list(self.qtarget.parameters())
        self.memory = defaults_to(self.memory, self._make_memory)

        self.root = nodes.ValueNode[Batch](None)
        qvalues = self._make_qvalue_prediction_node(self.root)
        qtargets_batch = self.root
        if self.ir_module is not None:
            self.ir_module.to(self.device)
            qtargets_batch = nodes.IR(self.ir_module, qtargets_batch)
        qtargets = self._make_targets_computation_node(qtargets_batch)
        self.loss = nodes.MSELoss(qvalues, qtargets, self.root)
        self.optimizer = self._make_optimizer()

    def to(self, device: torch.device) -> Self:
        self.qnetwork = self.qnetwork.to(device)
        self.qtarget = self.qtarget.to(device)
        if self.mixer is not None:
            self.mixer = self.mixer.to(device)
        if self.ir_module is not None:
            self.ir_module = self.ir_module.to(device)
        self.device = device

    def save(self, to_directory: str):
        torch.save(self.qnetwork.state_dict(), f"{to_directory}/qnetwork.pt")
        if self.mixer is not None:
            self.mixer.save(os.path.join(to_directory, "mixer"))
        if self.ir_module is not None:
            self.ir_module.save(os.path.join(to_directory, "ir"))

    def load(self, from_directory: str):
        self.qnetwork.load_state_dict(torch.load(f"{from_directory}/qnetwork.pt"))
        self.qtarget.load_state_dict(self.qnetwork.state_dict())
        if self.mixer is not None:
            self.mixer.load(os.path.join(from_directory, "mixer"))
        if self.ir_module is not None:
            self.ir_module.load(os.path.join(from_directory, "ir"))

    def _make_qvalue_prediction_node(self, batch: nodes.Node[Batch]) -> nodes.Node[torch.Tensor]:
        qvalues = nodes.QValues(self.qnetwork, batch)
        if self.mixer is not None:
            mixed_qvalues = nodes.QValueMixer(self.mixer.to(self.device), qvalues, batch)
            qvalues = mixed_qvalues
            self.parameters += list(self.mixer.parameters())
        return qvalues

    def _make_targets_computation_node(self, batch: nodes.Node[Batch]) -> nodes.Node[torch.Tensor]:
        if self.double_qlearning:
            next_qvalues = nodes.DoubleQLearning(self.qnetwork, self.qtarget, batch)
        else:
            next_qvalues = nodes.NextQValues(self.qtarget, batch)

        if self.mixer is not None:
            mixed_next_qvalues = nodes.QValueMixer(deepcopy(self.mixer).to(self.device), next_qvalues, batch)
            next_qvalues = mixed_next_qvalues
            self.target_parameters += list(mixed_next_qvalues.mixer.parameters())

        target_qvalues = nodes.Target(self.gamma, next_qvalues, batch)
        return target_qvalues

    def _make_memory(self) -> ReplayMemory:
        if self.update_frequency == "step":
            return TransitionMemory(50_000)
        return EpisodeMemory(50_000)

    def _make_optimizer(self):
        match self.optimizer_str:
            case "adam":
                return torch.optim.Adam(self.parameters, lr=self.lr)
            case "rmsprop":
                return torch.optim.RMSprop(self.parameters, lr=self.lr)
            case other:
                raise ValueError(f"Unknown optimizer: {other}")

    def update_episode(self, episode: Episode, episode_num: int):
        if self.update_frequency == "step":
            return
        raise NotImplementedError()

    def update_step(self, transition: Transition, step_num: int):
        if self.update_frequency == "episode":
            return
        self.memory.add(transition)
        if step_num % self.train_interval != 0 or len(self.memory) < self.batch_size:
            return
        self.root.value = self.memory.sample(self.batch_size).to(self.device)
        loss = self.loss.value
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._soft_update()

    def _soft_update(self):
        for param, target in zip(self.parameters, self.target_parameters):
            new_value = (1 - self.tau) * target.data + self.tau * param.data
            target.data.copy_(new_value, non_blocking=True)

    def show(self):
        """Display the computation graph"""
        import matplotlib.pyplot as plt
        import networkx as nx

        edges = []
        to_visit = set[nodes.Node]([self.root])
        visited = set[nodes.Node]()
        while len(to_visit) > 0:
            current = to_visit.pop()
            for child in current.children:
                edges.append((current, child))
                if child not in visited:
                    to_visit.add(child)
            visited.add(current)
        graph = nx.DiGraph()
        graph.add_edges_from(edges)
        pos = nodes.compute_positions(visited)
        # pos = nx.circular_layout(graph)
        # pos = nx.spectral_layout(graph)
        nx.draw_networkx(graph, pos)
        plt.savefig("graph.png")
        # plt.show()
