from dataclasses import KW_ONLY, dataclass
from typing import Literal

import torch
from torch.nn.utils.rnn import pack_padded_sequence

from marl.models import Batch

from ..dqn import DQN


@dataclass
class MITrainer(DQN):
    noise_size: int
    n_actions: int
    n_agents: int
    state_size: int
    state_extras_size: int
    _: KW_ONLY
    train_interval: tuple[int, Literal["episode"]] = (1, "episode")  # type: ignore
    mi_loss_coef: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        assert self.train_interval[1] == "episode", "MAVEN must be trained on full episodes to compute the MI loss on whole trajectories."
        self._mi_loss = torch.nn.CrossEntropyLoss(reduction="mean")
        self.trajectory_aggregator = TrajectoryAggregator(self.n_agents, self.n_actions, self.state_size, self.state_extras_size)
        self.discriminator = Discriminator(self.trajectory_aggregator.output_size, self.noise_size)
        settings = self.optimiser.param_groups[0].copy()
        settings.pop("params")
        self.optimiser.add_param_group({"params": list(self.trajectory_aggregator.parameters()), **settings})
        self.optimiser.add_param_group({"params": self.discriminator.parameters(), **settings})
        self.name = self.__class__.__name__
        if self.mixer is not None:
            self.name += f"-{self.mixer.name}"

    def get_mixing_kwargs(self, batch: Batch, all_qvalues: torch.Tensor, is_next: bool = True):
        return {"maven_noise": batch["maven-noise"]}

    def train(self, time_step: int, batch: Batch):
        qvalues, chosen_qvalues = self._compute_qvalues(batch)
        with torch.no_grad():
            qtargets = self._compute_qtargets(batch)
        td_loss, td_error = self._compute_td_loss(chosen_qvalues, qtargets, batch)
        maven_loss = self._compute_maven_loss(batch, qvalues)
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

    def _compute_maven_loss(self, batch: Batch, qvalues: torch.Tensor):
        assert qvalues.grad_fn is not None, "Gradient should flow through all_qvalues for the MI loss to work !"
        qvalues = qvalues.masked_fill(~batch.available_actions, -torch.inf)
        embeddings = self.trajectory_aggregator.forward(qvalues, batch.states, batch.states_extras, batch.masks)
        predicted_class = self.discriminator.forward(embeddings)
        # Retrieve the actual noise and transform from one-hot to class indices.
        noise = batch["maven-noise"][0]
        ground_truth = noise.argmax(dim=-1).long()
        mi_loss = self._mi_loss.forward(predicted_class, ground_truth)
        return self.mi_loss_coef * mi_loss


@dataclass(unsafe_hash=True)
class TrajectoryAggregator(torch.nn.Module):
    n_agents: int
    n_actions: int
    state_size: int
    state_extras_size: int
    output_size: int = 32

    def __post_init__(self):
        super().__init__()
        self.rnn = torch.nn.GRU(self.input_size, self.output_size)

    @property
    def input_size(self):
        return self.n_actions * self.n_agents + self.state_size + self.state_extras_size

    def forward(self, qvalues: torch.Tensor, states: torch.Tensor, states_extras: torch.Tensor, masks: torch.Tensor):
        soft_qvalues = torch.nn.functional.softmax(qvalues, dim=-1)
        soft_qvalues = soft_qvalues.flatten(2)
        trajectories = torch.cat([soft_qvalues, states, states_extras], dim=-1)
        ep_lengths = masks.sum(dim=0).long()
        trajectories = pack_padded_sequence(trajectories, ep_lengths.cpu(), enforce_sorted=False)
        _, last_hidden_states = self.rnn.forward(trajectories)
        # Embeddings (last_hidden states) have shape (1, batch_size, embedding_size)
        return last_hidden_states[0]


@dataclass(unsafe_hash=True)
class Discriminator(torch.nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)
