import random
from collections import deque
from dataclasses import KW_ONLY, dataclass, field
from typing import Literal

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence

from marl.agents.bandits import UniformOneHot
from marl.agents.hierarchical.maven_agent import MAVENAgent
from marl.models import Batch, EpisodeMemory

from ..dqn import DQN


@dataclass
class MAVEN(DQN[EpisodeMemory]):
    noise_size: int
    n_actions: int
    n_agents: int
    state_size: int
    state_extras_size: int
    _: KW_ONLY
    z_policy_type: Literal["uniform", "max-entropy", "return"]
    mi_loss_coef: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        match self.z_policy_type:
            case "uniform":
                self.z_bandit = UniformOneHot(self.noise_size)
            case "return":
                pass
            case "max-entropy":
                raise NotImplementedError("Max-entropy z policy is not implemented yet.")

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
        # Retrieve the noise from the transition details instead of parsing the state extras implicitly
        noise = batch["maven-noise"][0]
        state_extras = batch.states_extras[:, :, : -self.noise_size]
        ground_truth = noise.argmax(dim=-1).long()
        all_qvalues_masked = all_qvalues.masked_fill(~batch.available_actions, -torch.inf)
        embeddings = self.trajectory_aggregator.forward(all_qvalues_masked, batch.states, state_extras, batch.masks)
        predicted_class = self.discriminator.forward(embeddings)
        mi_loss = self._mi_loss.forward(predicted_class, ground_truth)
        return self.mi_loss_coef * mi_loss

    def make_agent(self) -> MAVENAgent:  # type: ignore[override]
        base_agent = super().make_agent()
        return MAVENAgent(
            noise_size=self.noise_size,
            workers=base_agent,
            z_policy=self.z_bandit,
        )

    def update_episode(self, episode, episode_num: int, time_step: int):
        logs = super().update_episode(episode, episode_num, time_step)
        if self.z_bandit is not None:
            # Compute total return of the episode
            import numpy as np

            episode_return = float(np.sum(episode.rewards))
            self.z_bandit.record_episode_return(episode_return)
        return logs


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


class UniformMAVENZPolicy(Categ):
    def _sample_episode_noise(self) -> np.ndarray:
        episode_noise = np.zeros(self.noise_size, dtype=np.float32)
        episode_noise[random.randint(0, self.noise_size - 1)] = 1.0
        return episode_noise


@dataclass
class ReturnsBanditMAVENZPolicy(MAVENZPolicy):
    noise_size: int
    learning_rate: float = 1e-2
    buffer_size: int = 512
    update_iters: int = 8
    batch_size: int = 64
    reward_scaling: float = 20.0
    epsilon: float = 0.1
    entropy_coef: float = 0.01
    device: torch.device | str = "cpu"
    _return_buffer: deque[tuple[int, float]] = field(init=False, repr=False)

    def __post_init__(self):
        super().__init__(self.noise_size)
        self.device = torch.device(self.device)
        self.logits = torch.nn.Parameter(torch.zeros(self.noise_size, device=self.device))
        self.optimiser = torch.optim.RMSprop([self.logits], lr=self.learning_rate)
        self._return_buffer = deque(maxlen=self.buffer_size)

    def _sample_episode_noise(self) -> np.ndarray:
        probabilities = torch.softmax(self.logits.detach(), dim=-1)
        if random.random() < self.epsilon:
            index = random.randint(0, self.noise_size - 1)
        else:
            index = int(torch.distributions.Categorical(probabilities).sample().item())

        episode_noise = np.zeros(self.noise_size, dtype=np.float32)
        episode_noise[index] = 1.0
        return episode_noise

    def record_episode_return(self, episode_return: float):
        if self._episode_noise is None:
            return

        episode_noise_index = int(np.argmax(self._episode_noise))
        self._return_buffer.append((episode_noise_index, float(episode_return)))
        if len(self._return_buffer) < self.batch_size:
            return

        for _ in range(self.update_iters):
            batch_indices = np.random.choice(len(self._return_buffer), size=self.batch_size, replace=False)
            sampled_indices = torch.tensor([self._return_buffer[index][0] for index in batch_indices], device=self.device)
            sampled_returns = torch.tensor([self._return_buffer[index][1] for index in batch_indices], device=self.device)
            sampled_returns = sampled_returns / self.reward_scaling
            sampled_returns = sampled_returns - sampled_returns.mean()

            log_probabilities = torch.log_softmax(self.logits, dim=-1)
            chosen_log_probs = log_probabilities.gather(0, sampled_indices)
            entropy = -(torch.softmax(self.logits, dim=-1) * log_probabilities).sum()
            loss = -(sampled_returns.detach() * chosen_log_probs).mean() - self.entropy_coef * entropy

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
