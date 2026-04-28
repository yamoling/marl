from dataclasses import KW_ONLY, dataclass

import torch
from torch.nn.utils.rnn import pack_padded_sequence

from marl.models import Batch, EpisodeMemory

from .dqn import DQN


@dataclass
class MAVEN(DQN[EpisodeMemory]):
    noise_size: int
    n_actions: int
    n_agents: int
    state_size: int
    state_extras_size: int
    _: KW_ONLY
    mi_loss_coef: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self._mi_loss = torch.nn.CrossEntropyLoss(reduction="mean")
        self.trajectory_aggregator = TrajectoryAggregator(
            self.n_agents, self.n_actions, self.state_size + self.state_extras_size - self.noise_size
        )
        self.discriminator = Discriminator(self.trajectory_aggregator.output_size, self.noise_size)
        settings = self.optimiser.param_groups[0].copy()
        settings.pop("params")
        self.optimiser.add_param_group({"params": list(self.trajectory_aggregator.parameters()), **settings})
        self.optimiser.add_param_group({"params": self.discriminator.parameters(), **settings})
        self.name = self.__class__.__name__
        if self.mixer is not None:
            self.name += f"-{self.mixer.name}"

    def train(self, time_step: int, batch: Batch):
        all_qvalues, qvalues = self._compute_qvalues(batch)
        with torch.no_grad():
            qtargets = self._compute_qtargets(batch)
        td_loss, td_error = self._compute_td_loss(qvalues, qtargets, batch)
        maven_loss = self._compute_maven_loss(batch, all_qvalues)
        loss = td_loss + maven_loss
        logs = {"td-loss": td_loss.item(), "maven-loss": maven_loss.item(), "loss": loss.item()}
        self.optimiser.zero_grad()
        loss.backward()
        if self.grad_norm_clipping is not None:
            logs["grad_norm"] = torch.nn.utils.clip_grad_norm_(self.target_updater.parameters, self.grad_norm_clipping).item()
        self.optimiser.step()
        logs = logs | self.memory.update(time_step, td_error=td_error)
        if self.vbe is not None:
            logs = logs | self.vbe.update(batch)
        return logs

    def _compute_maven_loss(self, batch: Batch, all_qvalues: torch.Tensor):
        assert all_qvalues.grad_fn is not None, "Gradient should flow through all_qvalues for the MI loss to work !"
        noise = batch.states_extras[0, :, -self.noise_size :]
        state_extras = batch.states_extras[:, :, : -self.noise_size]
        ground_truth = noise.argmax(dim=-1)
        all_qvalues[~batch.available_actions] = -torch.inf
        embeddings = self.trajectory_aggregator.forward(all_qvalues, batch.states, state_extras, batch.masks)
        predicted_class = self.discriminator.forward(embeddings)
        mi_loss = self._mi_loss.forward(predicted_class, ground_truth)
        return self.mi_loss_coef * mi_loss


@dataclass(unsafe_hash=True)
class TrajectoryAggregator(torch.nn.Module):
    n_agents: int
    n_actions: int
    state_size: int
    output_size: int = 32

    def __post_init__(self):
        super().__init__()
        self.rnn = torch.nn.GRU(self.input_size, self.output_size)

    @property
    def input_size(self):
        return self.n_actions * self.n_agents + self.state_size

    def forward(self, qvalues: torch.Tensor, states: torch.Tensor, state_extras: torch.Tensor, masks: torch.Tensor):
        soft_qvalues = torch.nn.functional.softmax(qvalues, dim=-1)
        soft_qvalues = soft_qvalues.flatten(2)
        trajectories = torch.cat([soft_qvalues, states, state_extras], dim=-1)
        ep_lengths = masks.sum(dim=0).long()
        trajectories = pack_padded_sequence(trajectories, ep_lengths.cpu(), enforce_sorted=False)
        _, last_hidden_states = self.rnn.forward(trajectories)
        # Embeddings (last_hidden states) have shape (1, batch_size, embedding_size)
        return last_hidden_states[0]


@dataclass(unsafe_hash=True)
class Discriminator(torch.nn.Module):
    """Classifier that takes as input an aggregated trajectory and outputs a class probability distribution across `noise_size` classes."""

    input_size: int
    noise_size: int
    _: KW_ONLY
    n_layers: int = 2
    hidden_size: int = 64

    def __post_init__(self):
        super().__init__()
        assert self.n_layers >= 2, "Discriminator must have at least 2 layers"
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
        )
        for _ in range(self.n_layers - 2):
            self.model.append(torch.nn.Linear(self.hidden_size, self.hidden_size))
            self.model.append(torch.nn.ReLU())
        self.model.append(torch.nn.Linear(self.hidden_size, self.noise_size))
        self.model.append(torch.nn.Softmax(dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)
