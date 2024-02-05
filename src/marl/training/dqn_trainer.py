from copy import deepcopy
from dataclasses import dataclass
from serde import serialize
from typing import Literal, Optional

import torch
from rlenv import Episode, Transition

from marl.intrinsic_reward import IRModule
from marl.models import Batch, ReplayMemory, Trainer
from marl.nn import LinearNN, RecurrentNN, NN
from marl.policy import Policy, EpsilonGreedy
from marl.qlearning.mixers import Mixer
from marl.training import nodes

from .qtarget_updater import TargetParametersUpdater, SoftUpdate


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
    grad_norm_clipping: Optional[float]

    def __init__(
        self,
        qnetwork: LinearNN | RecurrentNN,
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
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.ir_module = ir_module
        self.mixer = mixer
        self.double_qlearning = double_qlearning
        if target_updater is None:
            target_updater = SoftUpdate(1e-2)
        self.target_params_updater = target_updater
        self.grad_norm_clipping = grad_norm_clipping
        self.memory = memory

        self.parameters = list[torch.nn.Parameter]()
        self.target_parameters = list[torch.nn.Parameter]()

        self.batch, self.td_error, self.loss = self._make_graph()
        self.optimiser = self._make_optimizer(optimiser)
        self.update_num = 0
        self.device = qnetwork.device
        self.to(self.device)

    def _make_graph(self):
        """
        Constructs the computation graph for the DQN trainer.

        Returns:
            batch (nodes.ValueNode[Batch]): The input batch of samples.
            td_error (nodes.TDError): The TD error node.
            loss (nodes.MSELoss): The loss node.
        """
        batch = nodes.ValueNode[Batch](None)  # type: ignore
        qvalues = self._make_qvalue_prediction_node(batch)
        qtargets_batch = batch

        if self.ir_module is not None:
            qtargets_batch = nodes.IR(self.ir_module, qtargets_batch)
        qtargets = self._make_targets_computation_node(qtargets_batch)
        td_error = nodes.TDError(qvalues, qtargets)
        loss = nodes.MSELoss(td_error, batch)
        return batch, td_error, loss

    def to(self, device: torch.device):
        self.device = device
        self.batch.to(device)

    def randomize(self):
        self.batch.randomize()

    def _make_qvalue_prediction_node(self, batch: nodes.Node[Batch]) -> nodes.Node[torch.Tensor]:
        qvalues = nodes.QValues(self.qnetwork, batch)
        self.parameters += list(self.qnetwork.parameters())
        if self.mixer is not None:
            mixed_qvalues = nodes.QValueMixer(self.mixer, qvalues, batch)
            qvalues = mixed_qvalues
            self.parameters += list(self.mixer.parameters())
        return qvalues

    def _make_targets_computation_node(self, batch: nodes.Node[Batch]) -> nodes.Node[torch.Tensor]:
        # The qtarget network does not have to be an attribute of the class
        # because it is only used to compute the target qvalues and its parameters
        # are added to the list of target parameters.
        qtarget = deepcopy(self.qnetwork)
        self.target_parameters += list(qtarget.parameters())
        if self.double_qlearning:
            next_qvalues = nodes.DoubleQLearning(self.qnetwork, qtarget, batch)
        else:
            next_qvalues = nodes.NextValues(qtarget, batch)

        if self.mixer is not None:
            target_mixer = deepcopy(self.mixer)
            next_qvalues = nodes.TargetQValueMixer(target_mixer, next_qvalues, batch)
            self.target_parameters += list(target_mixer.parameters())

        target_qvalues = nodes.Target(self.gamma, next_qvalues, batch)
        return target_qvalues

    def _make_optimizer(self, optimiser: Literal["adam", "rmsprop"]):
        assert len(self.parameters) == len(self.target_parameters)
        for param, target_param in zip(self.parameters, self.target_parameters):
            assert param.shape == target_param.shape
        match optimiser:
            case "adam":
                return torch.optim.Adam(self.parameters, lr=self.lr)
            case "rmsprop":
                return torch.optim.RMSprop(self.parameters, lr=self.lr)
            case other:
                raise ValueError(f"Unknown optimizer: {other}")

    def _update(self, step_num: int) -> dict[str, float]:
        if len(self.memory) < self.batch_size:
            return {}
        self.update_num += 1
        if self.update_num % self.update_interval != 0:
            return {}

        self.batch.value = self.memory.sample(self.batch_size).to(self.device)
        loss = self.loss.value
        self.optimiser.zero_grad()
        loss.backward()
        log = {"loss": loss.item(), "td_error": self.td_error.value.mean().item()}
        if self.grad_norm_clipping is not None:
            log["grad_norm"] = torch.nn.utils.clip_grad.clip_grad_norm_(self.parameters, self.grad_norm_clipping).item()

        self.optimiser.step()
        self.policy.update(step_num)
        if isinstance(self.policy, EpsilonGreedy):
            log["epsilon"] = self.policy.epsilon.value
        if self.ir_module is not None:
            log["ir_loss"] = self.ir_module.update()
            log["ir"] = 0
        self.memory.update(self.batch.value, self.td_error.value.detach())
        self.target_params_updater.update(self.parameters, self.target_parameters)
        return log

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
        to_visit = set[nodes.Node]([self.batch])
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
