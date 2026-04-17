from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from marlenv import Episode, Transition
from marlenv.utils import Schedule

from marl.models import Batch, Mixer, Policy, ReplayMemory, Trainer
from marl.models.batch import EpisodeBatch
from marl.models.nn.options import OptionCriticNetwork
from marl.models.replay_memory import EpisodeMemory, TransitionMemory
from marl.policy import ArgMax, EpsilonGreedy
from marl.training.qtarget_updater import HardUpdate, TargetParametersUpdater


@dataclass
class PPOC(Trainer):
    r"""PPO-style Option-Critic for multi-agent environments.

    Mathematical mapping to Bacon et al. (2016):
    - Critic estimates $Q_\Omega(s, \omega)$ and bootstraps with $U(\omega, s')$.
    - Intra-option policy is optimized with a clipped PPO surrogate using
      advantages built from the option-value critic.
    - Termination update follows Theorem 2, using
      $A_\Omega(s', \omega) = Q_\Omega(s', \omega) - V_\Omega(s')$ at next state.
    """

    oc: OptionCriticNetwork
    n_agents: int
    mixer: Mixer | None = None
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    n_epochs: int = 10
    eps_clip: float = 0.2
    critic_c1: InitVar[Schedule | float] = 0.5
    c1: Schedule = field(init=False)
    entropy_c2: InitVar[Schedule | float] = 0.01
    c2: Schedule = field(init=False)
    termination_c3: float = 1.0
    termination_reg: float = 0.01
    train_interval: int = 64
    minibatch_size: int = 32
    normalize_advantages: bool = True
    grad_norm_clipping: float | None = None
    train_on: Literal["episode", "transition"] = "transition"
    q_updater: InitVar[TargetParametersUpdater | None] = None
    target_updater: TargetParametersUpdater = field(init=False)
    option_train_policy: Policy = field(default_factory=lambda: EpsilonGreedy.constant(0.1))
    optimizer: torch.optim.Optimizer = field(init=False)
    memory: ReplayMemory[Any, Any] = field(init=False)

    def __post_init__(
        self,
        critic_c1: Schedule | float,
        entropy_c2: Schedule | float,
        q_updater: TargetParametersUpdater | None,
    ):
        super().__init__()
        assert self.minibatch_size <= self.train_interval

        self.oc = self.oc.to(self.device)
        self.target_oc = deepcopy(self.oc)
        if self.mixer is not None:
            self.mixer = self.mixer.to(self.device)
            self.target_mixer = deepcopy(self.mixer)
            self.name = f"PPOC-{self.mixer.name}"
        else:
            self.target_mixer = None
            self.name = "PPOC"

        self._ratio_min = 1 - self.eps_clip
        self._ratio_max = 1 + self.eps_clip
        self._parameters = self.compute_parameters()
        self.optimizer = torch.optim.Adam(self._parameters, lr=self.lr)
        if isinstance(critic_c1, (float, int)):
            critic_c1 = Schedule.constant(critic_c1)
        self.c1 = critic_c1
        if isinstance(entropy_c2, (float, int)):
            entropy_c2 = Schedule.constant(entropy_c2)
        self.c2 = entropy_c2

        if self.train_on == "transition":
            self.memory = TransitionMemory(self.train_interval)
        else:
            self.memory = EpisodeMemory(self.train_interval)

        if q_updater is None:
            q_updater = HardUpdate(200)
        self.target_updater = q_updater
        self.target_updater.add_parameters(self.oc.parameters(), self.target_oc.parameters())
        if self.mixer is not None and self.target_mixer is not None:
            self.target_updater.add_parameters(self.mixer.parameters(), self.target_mixer.parameters())

    @property
    def n_options(self):
        return self.oc.n_options

    def compute_parameters(self):
        params = list(self.oc.parameters())
        if self.mixer is not None:
            params += list(self.mixer.parameters())
        return params

    def _compute_selected_q_options(self, batch: Batch, use_target: bool = False):
        options = batch["options"].long().unsqueeze(-1)
        net = self.target_oc if use_target else self.oc
        q_options = net.compute_q_options(batch.obs, batch.extras)
        q_options = torch.gather(q_options, dim=-1, index=options).squeeze(-1)
        if self.mixer is not None:
            assert self.mixer is not None
            mixer = self.target_mixer if use_target else self.mixer
            assert mixer is not None
            q_options = mixer.forward(q_options, batch.states)
        return q_options

    def _compute_value_on_arrival(self, batch: Batch):
        options = batch["options"].long().unsqueeze(-1)
        next_values = self.target_oc.value_on_arrival(batch.next_obs, batch.next_extras, options)
        if self.target_mixer is not None:
            next_values = self.target_mixer.forward(next_values, batch.next_states)
        return next_values

    def _compute_training_data(self, batch: Batch):
        with torch.no_grad():
            values = self._compute_selected_q_options(batch, use_target=False)
            next_values = self._compute_value_on_arrival(batch)

            values[batch.masked_indices] = 0.0
            next_values[batch.dones] = 0.0
            advantages = batch.compute_gae(
                self.gamma,
                values,
                next_values,
                trace_decay=self.gae_lambda,
                normalize=self.normalize_advantages,
            )
            advantages[batch.masked_indices] = 0.0
            returns = advantages + values
            returns[batch.masked_indices] = 0.0
        return returns, advantages

    def _compute_policy_terms(self, batch: Batch):
        actions = batch.actions.long()
        log_probs = torch.zeros_like(actions, dtype=torch.float32, device=self.device)
        entropies = torch.zeros_like(actions, dtype=torch.float32, device=self.device)

        leading_shape = actions.shape[:-1]
        for idx in np.ndindex(*leading_shape):
            options = batch["options"][idx].tolist()
            dist = self.oc.policy(batch.obs[idx], batch.extras[idx], batch.available_actions[idx], options)
            log_probs[idx] = dist.log_prob(actions[idx])
            entropies[idx] = dist.entropy()
        dist2 = self.oc.policy(batch.obs, batch.extras, batch.available_actions, batch["options"])
        log_probs2 = dist2.log_prob(actions)
        return log_probs, entropies

    def _expand_agent_tensor(self, tensor: torch.Tensor, n_agents: int):
        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[-1] == n_agents:
            return tensor
        return tensor.unsqueeze(-1).expand(*tensor.shape, n_agents)

    def _index_tuple(self, batch: Batch, indices: np.ndarray):
        if isinstance(batch, EpisodeBatch):
            return (slice(None), indices)
        return (indices,)

    def train(self, batch: Batch, step_num: int):
        self.c1.update(step_num)
        self.c2.update(step_num)

        if self.mixer is None:
            batch = batch.for_individual_learners()

        with torch.no_grad():
            old_log_probs, _ = self._compute_policy_terms(batch)
            returns, critic_advantages = self._compute_training_data(batch)

            if self.mixer is None:
                actor_advantages = critic_advantages
            else:
                actor_advantages = critic_advantages.repeat_interleave(batch.n_agents).view(*critic_advantages.shape, batch.n_agents)

        critic_losses: list[float] = []
        actor_losses: list[float] = []
        entropy_losses: list[float] = []
        termination_losses: list[float] = []
        losses: list[float] = []
        ratios: list[np.ndarray] = []
        entropies: list[np.ndarray] = []
        norms: list[float] = []

        for _ in range(self.n_epochs):
            indices = np.random.choice(batch.size, self.minibatch_size, replace=False)
            minibatch = batch.get_minibatch(indices).to(self.device)
            if self.mixer is None:
                minibatch = minibatch.for_individual_learners()
            idx = self._index_tuple(batch, indices)

            mini_returns = returns[idx]
            mini_actor_advantages = actor_advantages[idx]
            mini_old_log_probs = old_log_probs[idx]

            mini_values = self._compute_selected_q_options(minibatch, use_target=False)
            critic_mask = minibatch.masks
            critic_loss = torch.sum(((mini_values - mini_returns) ** 2) * critic_mask) / critic_mask.sum().clamp_min(1.0)

            mini_new_log_probs, mini_entropy = self._compute_policy_terms(minibatch)
            actor_mask = self._expand_agent_tensor(minibatch.masks, minibatch.n_agents)
            ratio = torch.exp(mini_new_log_probs - mini_old_log_probs)
            surrogate1 = ratio * mini_actor_advantages
            surrogate2 = torch.clamp(ratio, self._ratio_min, self._ratio_max) * mini_actor_advantages
            surr_min = torch.min(surrogate1, surrogate2)
            actor_loss = -torch.sum(surr_min * actor_mask) / actor_mask.sum().clamp_min(1.0)

            entropy_loss = -torch.sum(mini_entropy * actor_mask) / actor_mask.sum().clamp_min(1.0)

            options = minibatch["options"].long().unsqueeze(-1)
            next_termination_probs = self.oc.termination_probability(minibatch.next_obs, minibatch.next_extras, options)
            with torch.no_grad():
                next_q_options = self.target_oc.compute_q_options(minibatch.next_obs, minibatch.next_extras)
                next_q_max = next_q_options.max(dim=-1).values
                next_q_current = torch.gather(next_q_options, dim=-1, index=options).squeeze(-1)
                if self.target_mixer is not None:
                    next_q_max = self.target_mixer.forward(next_q_max, minibatch.next_states)
                    next_q_current = self.target_mixer.forward(next_q_current, minibatch.next_states)
                next_advantage = next_q_current - next_q_max

            termination_mask = self._expand_agent_tensor(minibatch.masks * minibatch.not_dones, minibatch.n_agents)
            termination_loss = torch.sum(
                next_termination_probs * (next_advantage + self.termination_reg) * termination_mask
            ) / termination_mask.sum().clamp_min(1.0)

            self.optimizer.zero_grad()
            loss = actor_loss + self.c1 * critic_loss + self.c2 * entropy_loss + self.termination_c3 * termination_loss
            loss.backward()
            if self.grad_norm_clipping is not None:
                norm = torch.nn.utils.clip_grad_norm_(self._parameters, self.grad_norm_clipping)
                norms.append(norm.detach().cpu().item())
            self.optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropy_losses.append(entropy_loss.item())
            termination_losses.append(termination_loss.item())
            losses.append(loss.item())
            ratios.append(ratio.detach().cpu().numpy())
            entropies.append(mini_entropy.detach().cpu().numpy())

        logs: dict[str, Any] = {
            "ppoc/min_critic_loss": min(critic_losses),
            "ppoc/max_critic_loss": max(critic_losses),
            "ppoc/mean_critic_loss": float(np.mean(critic_losses)),
            "ppoc/min_actor_loss": min(actor_losses),
            "ppoc/max_actor_loss": max(actor_losses),
            "ppoc/mean_actor_loss": float(np.mean(actor_losses)),
            "ppoc/min_entropy_loss": min(entropy_losses),
            "ppoc/max_entropy_loss": max(entropy_losses),
            "ppoc/mean_entropy_loss": float(np.mean(entropy_losses)),
            "ppoc/min_termination_loss": min(termination_losses),
            "ppoc/max_termination_loss": max(termination_losses),
            "ppoc/mean_termination_loss": float(np.mean(termination_losses)),
            "ppoc/min_loss": min(losses),
            "ppoc/max_loss": max(losses),
            "ppoc/mean_loss": float(np.mean(losses)),
            "ppoc/min_ratio": float(np.min(ratios)),
            "ppoc/max_ratio": float(np.max(ratios)),
            "ppoc/mean_ratio": float(np.mean(ratios)),
            "ppoc/mean_entropy": float(np.mean(entropies)),
            "ppoc/min_entropy": float(np.min(entropies)),
            "ppoc/max_entropy": float(np.max(entropies)),
            "ppoc/c1": self.c1.value,
            "ppoc/c2": self.c2.value,
            **self.option_train_policy.update(step_num),
            **self.target_updater.update(step_num),
        }
        if norms:
            logs["ppoc/mean_grad_norm"] = float(np.mean(norms))
            logs["ppoc/max_grad_norm"] = max(norms)
        return logs

    def update_step(self, transition: Transition, time_step: int):
        if not self.memory.update_on_transitions:
            return self.option_train_policy.update(time_step)
        self.memory.add(transition)
        logs = self.option_train_policy.update(time_step)
        if not self.memory.is_full:
            return logs
        batch = self.memory.as_batch().to(self.device)
        self.memory.clear()
        logs = logs | self.train(batch, time_step)
        return logs

    def update_episode(self, episode: Episode, episode_num: int, time_step: int):
        del episode_num
        if not self.memory.update_on_episodes:
            return self.option_train_policy.update(time_step)
        self.memory.add(episode)
        logs = self.option_train_policy.update(time_step)
        if not self.memory.is_full:
            return logs
        batch = self.memory.as_batch().to(self.device)
        self.memory.clear()
        logs = logs | self.train(batch, time_step)
        return logs

    def make_agent(self, test_policy: Policy | None = None):
        from marl.agents import OptionAgent

        if test_policy is None:
            test_policy = ArgMax()
        return OptionAgent(self.n_options, self.n_agents, self.oc, self.option_train_policy, test_policy)
