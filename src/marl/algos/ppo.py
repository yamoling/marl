from collections import defaultdict
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from marlenv import Episode, Transition

from marl.models import Batch, EpisodeMemory, Mixer, Trainer, TransitionMemory
from marl.models.batch import EpisodeBatch
from marl.models.nn import Actor, Critic
from marl.utils import Schedule


@dataclass
class PPO(Trainer):
    """Proximal Policy Optimization trainer. Either MAPPO (with a mixer) or IPPO (without mixer)."""

    actor: Actor
    critic: Critic
    mixer: Mixer | None
    _: KW_ONLY
    train_interval: tuple[int, Literal["step", "episode"]] = (64, "step")
    lr_actor: float = 5e-4
    lr_critic: float = 1e-3
    n_epochs: int = 20
    eps_clip: float = 0.2
    c1: Schedule = field(default_factory=lambda: Schedule.constant(0.5))
    c2: Schedule = field(default_factory=lambda: Schedule.constant(0.01))
    gae_lambda: float = 0.95
    minibatch_size: int = 32
    normalize_advantages: bool = True
    early_stopping_kl: float | None = None
    """Early stopping if the KL divergence between the old and new policy is higher than this threshold. If None, no early stopping is applied."""
    value_loss: Literal["huber", "mse"] = "huber"

    def __post_init__(self):
        super().__post_init__()
        match self.train_interval:
            case (n, "step"):
                if self.actor.is_recurrent:
                    raise ValueError("Recurrent neural networks should train on full episodes, not on transitions !")
                self.memory = TransitionMemory(n)
            case (n, "episode"):
                self.memory = EpisodeMemory(n)
        self.batch_size = n
        assert self.minibatch_size <= self.batch_size, (
            f"The batch size (i.e.: train_interval={self.batch_size}) should be greater than the minibatch size ({self.minibatch_size})."
        )
        self._ratio_min = 1 - self.eps_clip
        self._ratio_max = 1 + self.eps_clip
        self._parameters = [*self.actor.parameters(), *self.critic.parameters()]
        if self.mixer is not None:
            self._parameters += self.mixer.parameters()
        param_groups = self._compute_param_groups(self.lr_actor, self.lr_critic)
        self._optimizer = torch.optim.AdamW(param_groups, eps=1e-5)

    @property
    def name(self):
        if self.mixer is None:
            return "IPPO"
        return f"MAPPO-{self.mixer.name}"

    def _compute_param_groups(self, lr_actor: float, lr_critic: float):
        params = [
            {"params": self.actor.parameters(), "lr": lr_actor, "name": "actor parameters"},
            {"params": self.critic.parameters(), "lr": lr_critic, "name": "critic parameters"},
        ]
        if self.mixer is not None:
            params.append({"params": self.mixer.parameters(), "lr": lr_critic, "name": "mixer parameters"})
        return params

    def add_intrinsic_rewards(self, batch: Batch, time_step: int) -> dict[str, Any]:
        """
        Add the intrinsic rewards to the batch rewards (in place) and update the intrinsic reward module.

        # Returns
            dict[str, Any]: intrinsic-reward metrics to log.
        """
        if self.ir_module is None:
            return {}
        batch.rewards = batch.rewards + self.ir_module.compute(batch)
        return self.ir_module.update(batch, time_step)

    def _compute_training_data(self, batch: Batch):
        """Compute the returns, advantages and action log_probs according to the current policy"""
        values = self.critic.value(batch.obs, batch.extras)
        next_values = self.critic.value(batch.next_obs, batch.extras)
        if self.mixer is not None:
            values = self.mixer.forward(values, batch.states, batch.states_extras)
            next_values = self.mixer.forward(next_values, batch.next_states, batch.next_states_extras) * batch.not_dones
        values[batch.masked_indices] = 0.0
        next_values[batch.dones] = 0.0
        assert torch.all(next_values[batch.masked_indices] == 0.0)
        advantages = batch.compute_gae(
            self.gamma, values, next_values, self.gae_lambda, normalize=self.normalize_advantages
        )
        returns = batch.compute_mc_returns(self.gamma, next_values[-1])
        advantages[batch.masked_indices] = 0.0
        return returns, advantages

    def train(self, step_num: int):
        if not self.memory.is_full:
            return {}
        batch = self.memory.as_batch().to(self.device)
        self.memory.clear()
        self.c1.update(step_num)
        self.c2.update(step_num)
        if self.mixer is None:
            batch = batch.for_individual_learners()
        ir_logs = self.add_intrinsic_rewards(batch, step_num)
        with torch.no_grad():
            old_dist = self.actor.policy(batch.obs, batch.extras, available_actions=batch.available_actions)
            old_log_probs = old_dist.log_prob(batch.actions)
            old_log_probs[batch.masked_indices] = 0.0
            returns, advantages = self._compute_training_data(batch)
        if self.mixer is not None:
            # For IPPO, the advantages are already computed agent-wise.
            advantages = advantages.repeat_interleave(batch.n_agents).view(*advantages.shape, batch.n_agents)
        log_lists = defaultdict(list)
        early_stopped = False
        for _ in range(self.n_epochs):
            indices = np.random.choice(batch.size, self.minibatch_size, replace=False)
            minibatch = batch.get_minibatch(indices)
            if self.mixer is None:
                minibatch = minibatch.for_individual_learners()
            if isinstance(minibatch, EpisodeBatch):
                indices = (
                    slice(None),
                    indices,
                )  # The episode dimension come second in episode batches: (time, episode, ...)
            else:
                indices = (indices,)
            mini_returns = returns[*indices]
            mini_advantages = advantages[*indices, :]

            # Actor loss (ratio between the new and old policy):
            # L^CLIP(θ) = E[ min(r(θ)A, clip(r(θ), 1 − ε, 1 + ε)A) ] in PPO paper
            mini_policy = self.actor.policy(
                minibatch.obs,
                minibatch.extras,
                available_actions=minibatch.available_actions,
            )
            mini_new_log_probs: torch.Tensor = mini_policy.log_prob(minibatch.actions)
            mini_new_log_probs[minibatch.masked_indices] = 0.0
            log_ratio = mini_new_log_probs - old_log_probs[indices]
            ratio = torch.exp(log_ratio)

            with torch.no_grad():
                approx_kl_div = torch.mean((ratio - 1) - log_ratio).item()
                log_lists["approx-kl-divergence"].append(approx_kl_div)
            # KL divergence early stopping, cf Stable baselines implementation
            # https://github.com/DLR-RM/stable-baselines3/blob/08d984c3ee30093ea37409cf29cfb7efdd4bdcfd/stable_baselines3/ppo/ppo.py#L267
            if self.early_stopping_kl is not None and approx_kl_div > 1.5 * self.early_stopping_kl:
                early_stopped = True
                break

            surrogate1 = mini_advantages * ratio
            surrogate2 = torch.clamp(ratio, self._ratio_min, self._ratio_max) * mini_advantages
            surr_min = torch.min(surrogate1, surrogate2)
            actor_loss = -torch.sum(surr_min) / minibatch.n_items  # Minus sign to maximize the objective

            # Use the Monte Carlo estimate of returns as target values
            # L^VF(θ) = E[(V(s) - V_targ(s))^2] in PPO paper
            mini_values = self.critic.value(minibatch.obs, minibatch.extras)
            if self.mixer is not None:
                mini_values = self.mixer.forward(mini_values, minibatch.states, minibatch.states_extras)
            mini_values[minibatch.masked_indices] = 0.0
            if self.value_loss == "huber":
                # Same parameters as the MAPPO paper
                huber_loss = torch.nn.functional.huber_loss(mini_values, mini_returns, delta=10.0, reduction="none")
                critic_loss = torch.sum(huber_loss * minibatch.masks) / minibatch.n_items
            else:
                td_error = mini_values - mini_returns
                critic_loss = torch.sum(td_error**2) / minibatch.n_items

            # S[\pi_0](s_t) in the paper (equation (9))
            entropy = mini_policy.entropy()
            if self.mixer is not None:
                # Sum the agent dimension for the masking on the next line
                entropy = entropy.sum(-1)
            entropy = entropy * minibatch.masks
            entropy_loss = -torch.sum(entropy) / minibatch.n_items  # Minus sign to maximize the entropy

            self._optimizer.zero_grad()
            # Equation (9) in the paper
            loss = actor_loss + self.c1 * critic_loss + self.c2 * entropy_loss
            loss.backward()
            if self.grad_norm_clipping is not None:
                norm = torch.nn.utils.clip_grad_norm_(self._parameters, self.grad_norm_clipping)
                log_lists["norms"].append(norm.detach().cpu().item())
            self._optimizer.step()
            log_lists["actor_loss"].append(actor_loss.item())
            log_lists["critic_loss"].append(critic_loss.item())
            log_lists["entropy_loss"].append(entropy_loss.item())
            log_lists["loss"].append(loss.item())
            log_lists["ratios"].append(ratio.detach().cpu().numpy())
            log_lists["entropies"].append(entropy.detach().cpu().numpy())
        return {
            **ir_logs,
            "early_stopped": early_stopped,
            "ppoc/c1": self.c1.value,
            "ppoc/c2": self.c2.value,
            **{f"ppoc/mean-{key}": np.mean(values) for key, values in log_lists.items()},
            **{f"ppoc/max-{key}": np.max(values) for key, values in log_lists.items()},
            **{f"ppoc/min-{key}": np.min(values) for key, values in log_lists.items()},
        }

    def update_step(self, transition: Transition, time_step: int):
        if not isinstance(self.memory, TransitionMemory):
            return {}
        self.memory.add(transition)
        return self.train(time_step)

    def update_episode(self, episode: Episode, episode_num: int, time_step: int):
        if not isinstance(self.memory, EpisodeMemory):
            return {}
        self.memory.add(episode)
        return self.train(time_step)

    def make_agent(self):
        from marl.agents import SimpleAgent

        return SimpleAgent(self.actor)
