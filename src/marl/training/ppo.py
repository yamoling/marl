import os
from collections import defaultdict
from dataclasses import KW_ONLY, InitVar, dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from marlenv import Episode, Transition
from marlenv.utils import Schedule

from marl.models import Batch, Mixer, ReplayMemory, Trainer
from marl.models.batch import EpisodeBatch
from marl.models.nn import ActorCritic, IRModule


@dataclass
class PPO[B: Batch](Trainer):
    """Proximal Policy Optimization trainer. Either MAPPO (with a mixer) or IPPO (without mixer)."""

    actor_critic: ActorCritic
    mixer: Mixer | None
    _: KW_ONLY
    ir_module: IRModule | None = None
    gamma: float = 0.99
    lr_actor: float = 5e-4
    lr_critic: float = 1e-3
    n_epochs: int = 20
    eps_clip: float = 0.2
    critic_c1: InitVar[Schedule | float] = 0.5
    c1: Schedule = field(init=False)
    entropy_c2: InitVar[Schedule | float] = 0.01
    c2: Schedule = field(init=False)
    train_interval: int = 64
    gae_lambda: float = 0.95
    grad_norm_clipping: float | None = None
    minibatch_size: int = 32
    normalize_advantages: bool = True
    optimizer: torch.optim.Optimizer = field(init=False)
    train_on: Literal["episode", "transition"] = "transition"
    memory: ReplayMemory[Any, B] = field(init=False)
    early_stopping_kl: float | None = None
    """Early stopping if the KL divergence between the old and new policy is higher than this threshold. If None, no early stopping is applied."""
    value_loss: Literal["huber", "mse"] = "huber"

    def __post_init__(self, critic_c1: Schedule | float, entropy_c2: Schedule | float):
        super().__init__()
        assert self.minibatch_size <= self.train_interval
        self.actor_critic = self.actor_critic
        if self.mixer is not None:
            self.mixer = self.mixer
            self.name = f"MAPPO-{self.mixer.name}"
        else:
            self.name = "IPPO"
        self._ratio_min = 1 - self.eps_clip
        self._ratio_max = 1 + self.eps_clip
        self._parameters = list(self.actor_critic.parameters())
        param_groups = self._compute_param_groups(self.lr_actor, self.lr_critic)
        self.optimizer = torch.optim.AdamW(param_groups, eps=1e-5)
        if isinstance(critic_c1, (float, int)):
            critic_c1 = Schedule.constant(critic_c1)
        self.c1 = critic_c1
        if isinstance(entropy_c2, (float, int)):
            entropy_c2 = Schedule.constant(entropy_c2)
        self.c2 = entropy_c2
        if self.train_on == "transition":
            from marl.models.replay_memory import TransitionMemory

            self.memory = TransitionMemory(self.train_interval)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            from marl.models.replay_memory import EpisodeMemory

            self.memory = EpisodeMemory(self.train_interval)  # pyright: ignore[reportAttributeAccessIssue]
        if self.actor_critic.is_recurrent and not self.train_on == "episode":
            raise ValueError("Recurrent neural networks should train on full episodes, not on transaitions !")

    def _compute_param_groups(self, lr_actor: float, lr_critic: float):
        params = [
            {"params": self.actor_critic.policy_parameters, "lr": lr_actor, "name": "actor parameters"},
            {"params": self.actor_critic.value_parameters, "lr": lr_critic, "name": "critic parameters"},
        ]
        if self.mixer is not None:
            params.append({"params": list(self.mixer.parameters()), "lr": lr_critic, "name": "mixer parameters"})
        return params

    def _compute_training_data(self, batch: Batch):
        """Compute the returns, advantages and action log_probs according to the current policy"""
        values = self.actor_critic.value(batch.obs, batch.extras)
        next_values = self.actor_critic.value(batch.next_obs, batch.extras)
        if self.mixer is not None:
            values = self.mixer.forward(values, batch.states)
            next_values = self.mixer.forward(next_values, batch.next_states) * batch.not_dones
        values[batch.masked_indices] = 0.0
        next_values[batch.dones] = 0.0
        assert torch.all(next_values[batch.masked_indices] == 0.0)
        advantages = batch.compute_gae(self.gamma, values, next_values, self.gae_lambda, normalize=self.normalize_advantages)
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
        if self.ir_module is not None:
            batch.rewards = batch.rewards + self.ir_module.compute(batch)
        with torch.no_grad():
            old_dist = self.actor_critic.policy(batch.obs, batch.extras, batch.available_actions)
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
                indices = (slice(None), indices)  # The episode dimension come second in episode batches: (time, episode, ...)
            else:
                indices = (indices,)
            mini_returns = returns[*indices]
            mini_advantages = advantages[*indices, :]

            # Actor loss (ratio between the new and old policy):
            # L^CLIP(θ) = E[ min(r(θ)A, clip(r(θ), 1 − ε, 1 + ε)A) ] in PPO paper
            mini_policy = self.actor_critic.policy(minibatch.obs, minibatch.extras, minibatch.available_actions)
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
            actor_loss = -torch.sum(surr_min) / minibatch.masks_sum  # Minus sign to maximize the objective

            # Use the Monte Carlo estimate of returns as target values
            # L^VF(θ) = E[(V(s) - V_targ(s))^2] in PPO paper
            mini_values = self.actor_critic.value(minibatch.obs, minibatch.extras)
            if self.mixer is not None:
                mini_values = self.mixer.forward(mini_values, batch.states)
            mini_values[minibatch.masked_indices] = 0.0
            if self.value_loss == "huber":
                # Same parameters as the MAPPO paper
                huber_loss = torch.nn.functional.huber_loss(mini_values, mini_returns, delta=10.0, reduction="none")
                critic_loss = torch.sum(huber_loss * minibatch.masks) / minibatch.masks_sum
            else:
                td_error = mini_values - mini_returns
                critic_loss = torch.sum(td_error**2) / minibatch.masks_sum

            # S[\pi_0](s_t) in the paper (equation (9))
            entropy = mini_policy.entropy()
            if self.mixer is not None:
                # Sum the agent dimension for the masking on the next line
                entropy = entropy.sum(-1)
            entropy = entropy * minibatch.masks
            entropy_loss = -torch.sum(entropy) / minibatch.masks_sum  # Minus sign to maximize the entropy

            self.optimizer.zero_grad()
            # Equation (9) in the paper
            loss = actor_loss + self.c1 * critic_loss + self.c2 * entropy_loss
            loss.backward()
            if self.grad_norm_clipping is not None:
                norm = torch.nn.utils.clip_grad_norm_(self._parameters, self.grad_norm_clipping)
                log_lists["norms"].append(norm.detach().cpu().item())
            self.optimizer.step()
            log_lists["actor_loss"].append(actor_loss.item())
            log_lists["critic_loss"].append(critic_loss.item())
            log_lists["entropy_loss"].append(entropy_loss.item())
            log_lists["loss"].append(loss.item())
            log_lists["ratios"].append(ratio.detach().cpu().numpy())
            log_lists["entropies"].append(entropy.detach().cpu().numpy())
        return {
            "early_stopped": early_stopped,
            "ppoc/c1": self.c1.value,
            "ppoc/c2": self.c2.value,
            **{f"ppoc/mean-{key}": np.mean(values) for key, values in log_lists.items()},
            **{f"ppoc/max-{key}": np.max(values) for key, values in log_lists.items()},
            **{f"ppoc/min-{key}": np.min(values) for key, values in log_lists.items()},
        }

    def update_step(self, transition: Transition, time_step: int):
        if not self.memory.update_on_transitions:
            return {}
        self.memory.add(transition)
        return self.train(time_step)

    def update_episode(self, episode: Episode, episode_num: int, time_step: int):
        if not self.memory.update_on_episodes:
            return {}
        self.memory.add(episode)
        return self.train(time_step)

    def make_agent(self):
        from marl.agents import SimpleActor

        return SimpleActor(self.actor_critic)

    @property
    def device(self):
        return self._device

    def save(self, directory_path: str):
        os.makedirs(directory_path, exist_ok=True)
        filename = os.path.join(directory_path, "actor_critic.weights")
        with open(filename, "wb") as f:
            torch.save(self.actor_critic.state_dict(), f)

    def load(self, directory_path: str):
        filename = os.path.join(directory_path, "actor_critic.weights")
        with open(filename, "rb") as f:
            self.actor_critic.load_state_dict(torch.load(f))
