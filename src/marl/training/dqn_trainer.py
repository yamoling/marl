from copy import deepcopy
from dataclasses import dataclass
from serde import serialize
from abc import ABC, abstractmethod
from typing import Literal, Optional

import os
import torch
from rlenv import Episode, Transition

from marl.intrinsic_reward import IRModule
from marl.models import Batch, ReplayMemory, Trainer
from marl.nn import LinearNN, RecurrentNN, NN
from marl.policy import Policy
from marl.qlearning.mixers import Mixer
from marl.training import nodes


@dataclass
class TargetParametersUpdater(ABC):
    name: str

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def update(self, current_params: list[torch.nn.Parameter], target_params: list[torch.nn.Parameter]):
        """Update the target network parameters based on the current network parameters"""


@dataclass
class HardUpdate(TargetParametersUpdater):
    update_period: int

    def __init__(self, update_period: int):
        super().__init__()
        self.update_period = update_period
        self.update_num = 0

    def update(self, current_params: list[torch.nn.Parameter], target_params: list[torch.nn.Parameter]):
        self.update_num += 1
        if self.update_num % self.update_period == 0:
            for param, target in zip(current_params, target_params):
                target.data.copy_(param.data, non_blocking=True)


@dataclass
class SoftUpdate(TargetParametersUpdater):
    tau: float

    def __init__(self, tau: float):
        super().__init__()
        self.tau = tau

    def update(self, current_params: list[torch.nn.Parameter], target_params: list[torch.nn.Parameter], time_step: int):
        for param, target in zip(current_params, target_params):
            new_value = (1 - self.tau) * target.data + self.tau * param.data
            target.data.copy_(new_value, non_blocking=True)


@serialize
@dataclass
class DQNTrainer(Trainer):
    qnetwork: NN
    policy: Policy
    memory: ReplayMemory
    gamma: float
    batch_size: int
    lr: float
    target_params_updater: TargetParametersUpdater
    double_qlearning: bool
    mixer: Optional[Mixer]
    ir_module: Optional[IRModule]

    def __init__(
        self,
        qnetwork: LinearNN | RecurrentNN,
        train_policy: Policy,
        memory: ReplayMemory,
        gamma: float = 0.99,
        batch_size: int = 64,
        update_interval: int = 5,
        lr: float = 1e-4,
        optimiser: Literal["adam", "rmsprop"] = "adam",
        target_update: TargetParametersUpdater = SoftUpdate(0.01),
        double_qlearning: bool = False,
        mixer: Optional[Mixer] = None,
        train_every: Literal["step", "episode"] = "step",
        ir_module: Optional[IRModule] = None,
    ):
        super().__init__(train_every, update_interval)
        self.qnetwork = qnetwork
        self.qtarget = deepcopy(self.qnetwork)
        self.policy = train_policy
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.ir_module = ir_module
        self.mixer = mixer
        self.target_mixer = deepcopy(self.mixer)
        self.double_qlearning = double_qlearning
        self.target_params_updater = target_update
        self.memory = memory
        self.device = qnetwork.device

        self.parameters = list(self.qnetwork.parameters())
        self.target_parameters = list(self.qtarget.parameters())

        self.root, self.loss = self._make_graph()
        self.optimiser = self._make_optimizer(optimiser)
        self.update_num = 0

    def _make_graph(self):
        batch = nodes.ValueNode[Batch](None)  # type: ignore
        qvalues = self._make_qvalue_prediction_node(batch)
        qtargets_batch = batch

        if self.ir_module is not None:
            qtargets_batch = nodes.IR(self.ir_module, qtargets_batch)
        qtargets = self._make_targets_computation_node(qtargets_batch)
        loss = nodes.MSELoss(qvalues, qtargets, batch)
        return batch, loss

    def to(self, device: torch.device):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.nn.Module):
                value.to(device, non_blocking=True)
        self.device = device
        return self

    def randomize(self):
        self.qnetwork.randomize()
        self.qtarget.randomize()
        if self.ir_module is not None:
            self.ir_module.randomize()
        if self.mixer is not None:
            self.mixer.randomize()

    def save(self, to_directory: str):
        torch.save(self.qnetwork.state_dict(), f"{to_directory}/qnetwork.weights")
        if self.mixer is not None:
            self.mixer.save(os.path.join(to_directory, "mixer"))
        if self.ir_module is not None:
            self.ir_module.save(os.path.join(to_directory, "ir"))

    def load(self, from_directory: str):
        self.qnetwork.load_state_dict(torch.load(f"{from_directory}/qnetwork.weights"))
        self.qtarget.load_state_dict(self.qnetwork.state_dict())
        if self.mixer is not None:
            self.mixer.load(os.path.join(from_directory, "mixer"))
        if self.ir_module is not None:
            self.ir_module.load(os.path.join(from_directory, "ir"))

    def _make_qvalue_prediction_node(self, batch: nodes.Node[Batch]) -> nodes.Node[torch.Tensor]:
        qvalues = nodes.QValues(self.qnetwork, batch)
        if self.mixer is not None:
            mixed_qvalues = nodes.QValueMixer(self.mixer, qvalues, batch)
            qvalues = mixed_qvalues
            self.parameters += list(self.mixer.parameters())
        return qvalues

    def _make_targets_computation_node(self, batch: nodes.Node[Batch]) -> nodes.Node[torch.Tensor]:
        if self.double_qlearning:
            next_qvalues = nodes.DoubleQLearning(self.qnetwork, self.qtarget, batch)
        else:
            next_qvalues = nodes.NextQValues(self.qtarget, batch)

        if self.target_mixer is not None:
            next_qvalues = nodes.QValueMixer(self.target_mixer, next_qvalues, batch)
            self.target_parameters += list(self.target_mixer.parameters())

        target_qvalues = nodes.Target(self.gamma, next_qvalues, batch)
        return target_qvalues

    def _make_optimizer(self, optimiser: Literal["adam", "rmsprop"]):
        match optimiser:
            case "adam":
                return torch.optim.Adam(self.parameters, lr=self.lr)
            case "rmsprop":
                return torch.optim.RMSprop(self.parameters, lr=self.lr)
            case other:
                raise ValueError(f"Unknown optimizer: {other}")

    def update(self, time_step: int):
        if len(self.memory) < self.batch_size:
            return
        self.update_num += 1
        if self.update_num % self.update_interval != 0:
            return
        self.policy.update(time_step)
        self.root.value = self.memory.sample(self.batch_size).to(self.device)
        loss = self.loss.value
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        self.target_params_updater.update(self.parameters, self.target_parameters)

    def update_episode(self, episode: Episode, episode_num: int, time_step: int):
        if not self.update_on_episodes:
            return
        self.memory.add(episode)
        self.update(time_step)

    def update_step(self, transition: Transition, step_num: int):
        if not self.update_on_steps:
            return
        self.memory.add(transition)
        self.update(step_num)

    def show(self, filename: str = "trainer.png"):
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
        plt.savefig(filename)
        # plt.show()
