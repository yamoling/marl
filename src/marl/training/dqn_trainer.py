from copy import deepcopy
from dataclasses import dataclass
from serde import serialize
from typing import Literal, Optional

import torch
from rlenv import Episode, Transition

from marl.intrinsic_reward import IRModule
from marl.models import Updatable, Batch, ReplayMemory, PrioritizedMemory, LinearNN, RecurrentNN, Policy, Mixer, QNetwork
from marl.models.trainer import Trainer
from marl.training import nodes
from marl.training.nodes import Node

from .qtarget_updater import TargetParametersUpdater, SoftUpdate


@serialize
@dataclass
class DQNTrainer(Trainer):
    qnetwork: QNetwork
    policy: Policy
    memory: ReplayMemory
    gamma: float
    batch_size: int
    lr: float
    target_params_updater: TargetParametersUpdater
    double_qlearning: bool
    mixer: Optional[Mixer]
    ir_module: Optional[IRModule]
    grad_norm_clipping: Optional[float]
    optimiser: Literal["adam", "rmsprop"]

    def __init__(
        self,
        qnetwork: QNetwork,
        train_policy: Policy,
        memory: ReplayMemory,
        gamma: float = 0.99,
        batch_size: int = 64,
        lr: float = 1e-4,
        optimiser: Literal["adam", "rmsprop"] = "adam",
        target_updater: Optional[TargetParametersUpdater] = None,
        double_qlearning: bool = False,
        mixer: Optional[Mixer] = None,
        train_interval: tuple[int, Literal["step", "episode"]] = (5, "step"),
        ir_module: Optional[IRModule] = None,
        grad_norm_clipping: Optional[float] = None,
    ):
        super().__init__(train_interval[1], train_interval[0])
        self.qnetwork = qnetwork
        self.policy = train_policy
        self.memory = memory
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        if target_updater is None:
            target_updater = SoftUpdate(1e-2)
        self.target_params_updater = target_updater
        self.double_qlearning = double_qlearning
        self.mixer = mixer
        self.ir_module = ir_module
        self.grad_norm_clipping = grad_norm_clipping
        self.optimiser = optimiser

        self.roots = list[nodes.Node]()
        # Every updatable object should be in this list.
        # Some nodes of the graph also need to be updated, so they are also in this list.
        self.updatables = list[Updatable]([train_policy, target_updater])
        self.updatables += self._make_graph(qnetwork.device)
        # IR module must be added after the BackProp node otherwise the IR is not yet computed.
        if self.ir_module is not None:
            self.updatables.append(self.ir_module)
        self.update_num = 0

    def _make_graph(self, device: torch.device):
        """
        Constructs the computation graph for the DQN trainer.

        Returns:
            batch (nodes.ValueNode[Batch]): The input batch of samples.
            td_error (nodes.TDError): The TD error node.
            loss (nodes.MSELoss): The loss node.
        """
        updatables = list[Updatable]()
        if isinstance(self.memory, PrioritizedMemory):
            batch = nodes.PERNode(self.memory, self.batch_size, device)
            updatables.append(batch)
        else:
            batch = nodes.MemoryNode(self.memory, self.batch_size, device)
        self.roots.append(batch)
        qvalues, parameters = self._make_qvalue_prediction_node(batch)
        qtargets_batch = batch

        if self.ir_module is not None:
            qtargets_batch = nodes.IR(self.ir_module, qtargets_batch)
        qtargets, target_parameters = self._make_targets_computation_node(qtargets_batch)

        # Add parameters and target parameters to the target parameters updater
        self.target_params_updater.add_parameters(parameters, target_parameters)

        td_error = nodes.TDError(qvalues, qtargets)
        # Don't forget to set the td_error node in the PERNode !
        if isinstance(batch, nodes.PERNode):
            batch.set_td_error_node(td_error)
        loss = nodes.MSELoss(td_error, batch)
        optimiser = self._make_optimiser(parameters, target_parameters)
        updatables.append(nodes.BackpropNode(loss, parameters, optimiser, self.grad_norm_clipping))
        return updatables

    def to(self, device: torch.device):
        for root in self.roots:
            root.to(device)
        return self

    def randomize(self):
        for root in self.roots:
            root.randomize()

    def _make_qvalue_prediction_node(
        self,
        batch: nodes.Node[Batch],
    ) -> tuple[Node[torch.Tensor], list[torch.nn.Parameter]]:
        qvalues = nodes.QValues(self.qnetwork, batch)
        parameters = list(self.qnetwork.parameters())
        if self.mixer is not None:
            mixed_qvalues = nodes.QValueMixer(self.mixer, qvalues, batch)
            qvalues = mixed_qvalues
            parameters += list(self.mixer.parameters())
        return qvalues, parameters

    def _make_targets_computation_node(
        self,
        batch: nodes.Node[Batch],
    ) -> tuple[Node[torch.Tensor], list[torch.nn.Parameter]]:
        # The qtarget network does not have to be an attribute of the class
        # because it is only used to compute the target qvalues and its parameters
        # are added to the list of target parameters.
        qtarget = deepcopy(self.qnetwork)
        target_parameters = list(qtarget.parameters())
        if self.double_qlearning:
            next_qvalues = nodes.DoubleQLearning(self.qnetwork, qtarget, batch)
        else:
            next_qvalues = nodes.NextValues(qtarget, batch)

        if self.mixer is not None:
            target_mixer = deepcopy(self.mixer)
            next_qvalues = nodes.TargetQValueMixer(target_mixer, next_qvalues, batch)
            target_parameters += list(target_mixer.parameters())

        target_qvalues = nodes.Target(self.gamma, next_qvalues, batch)
        return target_qvalues, target_parameters

    def _make_optimiser(
        self,
        parameters: list[torch.nn.Parameter],
        target_parameters: list[torch.nn.Parameter],
    ):
        assert len(parameters) == len(target_parameters)
        for param, target_param in zip(parameters, target_parameters):
            assert param.shape == target_param.shape
        match self.optimiser:
            case "adam":
                return torch.optim.Adam(parameters, lr=self.lr)
            case "rmsprop":
                return torch.optim.RMSprop(parameters, lr=self.lr)
            case other:
                raise ValueError(f"Unknown optimizer: {other}")

    def _update(self, step_num: int) -> dict[str, float]:
        if len(self.memory) < self.batch_size:
            return {}
        self.update_num += 1
        if self.update_num % self.update_interval != 0:
            return {}
        # Invalidate the current batch to re-sample a new one from the memory, etc
        for root in self.roots:
            root.invalidate_value()
        logs = {}
        for updatable in self.updatables:
            logs.update(updatable.update(step_num))
        return logs

    def update_episode(self, episode: Episode, episode_num: int, time_step: int) -> dict[str, float]:
        if not self.update_on_episodes:
            return {}
        self.memory.add(episode)
        return self._update(time_step)

    def update_step(self, transition: Transition, time_step: int) -> dict[str, float]:
        if not self.update_on_steps:
            return {}
        self.memory.add(transition)
        return self._update(time_step)

    def show(self, filename: str = "trainer.png"):
        """Display the computation graph"""
        import matplotlib.pyplot as plt
        import networkx as nx

        edges = []
        to_visit = set[nodes.Node](self.roots)
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
