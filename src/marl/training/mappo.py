import os
from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from marlenv import Episode, Transition
from marlenv.utils import Schedule

from marl.models import Batch, Mixer, ReplayMemory, Trainer
from marl.models.batch import EpisodeBatch
from marl.models.nn import ActorCritic, IRModule


@dataclass
class MAPPO[B: Batch](Trainer):
    """Multi-Agent Proximal Policy Optimization"""

    actor_critic: ActorCritic
    mixer: Mixer | None
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

    def __post_init__(self, critic_c1: Schedule | float, entropy_c2: Schedule | float):
        super().__init__(torch.device("cpu"))
        assert self.minibatch_size <= self.train_interval
        self.actor_critic = self.actor_critic.to(self.device)
        if self.mixer is not None:
            self.mixer = self.mixer.to(self.device)
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
        returns = batch.compute_mc_returns(self.gamma, next_values)
        advantages[batch.masked_indices] = 0.0
        assert torch.all(advantages[batch.masked_indices] == 0)
        assert torch.all(returns[batch.masked_indices] == 0.0)
        return returns, advantages

    def train(self, batch: Batch, step_num: int):
        self.c1.update(step_num)
        self.c2.update(step_num)
        if self.mixer is None:
            batch = batch.for_individual_learners()
        if self.ir_module is not None:
            batch.rewards = batch.rewards + self.ir_module.compute(batch)

        old_ac = deepcopy(self.actor_critic)
        with torch.no_grad():
            returns, advantages = self._compute_training_data(batch)
        if self.mixer is not None:
            # For IPPO, the advantages are already computed agent-wise.
            advantages = advantages.repeat_interleave(batch.n_agents).view(*advantages.shape, batch.n_agents)
        critic_losses, actor_losses, entropy_losses, losses, ratios, entropies, norms = [], [], [], [], [], [], []
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
            with torch.no_grad():
                dist = old_ac.policy(minibatch.obs, minibatch.extras, minibatch.available_actions)
                mini_log_probs = dist.log_prob(minibatch.actions)
                mini_log_probs[minibatch.masked_indices] = 0.0

            # Use the Monte Carlo estimate of returns as target values
            # L^VF(θ) = E[(V(s) - V_targ(s))^2] in PPO paper
            mini_values = self.actor_critic.value(minibatch.obs, minibatch.extras)
            if self.mixer is not None:
                mini_values = self.mixer.forward(mini_values, batch.states)
            mini_values[minibatch.masked_indices] = 0.0
            td_error = mini_values - mini_returns
            critic_loss = torch.sum(td_error**2) / minibatch.masks_sum

            # Actor loss (ratio between the new and old policy):
            # L^CLIP(θ) = E[ min(r(θ)A, clip(r(θ), 1 − ε, 1 + ε)A) ] in PPO paper
            mini_policy = self.actor_critic.policy(minibatch.obs, minibatch.extras, minibatch.available_actions)
            mini_new_log_probs: torch.Tensor = mini_policy.log_prob(minibatch.actions)
            mini_new_log_probs[minibatch.masked_indices] = 0.0
            ratio = torch.exp(mini_new_log_probs - mini_log_probs)
            surrogate2 = torch.clamp(ratio, self._ratio_min, self._ratio_max) * mini_advantages

            surrogate1 = mini_advantages * ratio
            surr_min = torch.min(surrogate1, surrogate2)
            actor_loss = -torch.sum(surr_min) / minibatch.masks_sum  # Minus sign to maximize the objective

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
                norms.append(norm.cpu().item())
            self.optimizer.step()
            critic_losses.append(critic_loss.item())
            actor_losses.append(actor_loss.item())
            entropy_losses.append(entropy_loss.item())
            losses.append(loss.item())
            ratios.append(ratio.numpy(force=True))
            entropies.append(entropy.numpy(force=True))
        return {
            "ppo/min_critic_loss": min(critic_losses),
            "ppo/max_critic_loss": max(critic_losses),
            "ppo/mean_critic_loss": np.mean(critic_losses).item(),
            "ppo/min_actor_loss": min(actor_losses),
            "ppo/max_actor_loss": max(actor_losses),
            "ppo/mean_actor_loss": np.mean(actor_losses).item(),
            "ppo/min_entropy_loss": min(entropy_losses),
            "ppo/max_entropy_loss": max(entropy_losses),
            "ppo/mean_entropy_loss": np.mean(entropy_losses).item(),
            "ppo/min_loss": min(losses),
            "ppo/max_loss": max(losses),
            "ppo/mean_loss": np.mean(losses).item(),
            "ppo/min_ratio": np.min(ratios).item(),
            "ppo/max_ratio": np.max(ratios).item(),
            "ppo/mean_ratio": np.mean(ratios).item(),
            "ppo/mean_entropy": np.mean(entropies).item(),
            "ppo/min_entropy": np.min(entropies).item(),
            "ppo/max_entropy": np.max(entropies).item(),
            "ppo/c1": self.c1.value,
            "ppo/c2": self.c2.value,
        }

    def update_step(self, transition: Transition, time_step: int):
        if not self.memory.update_on_transitions:
            return {}
        self.memory.add(transition)
        if not self.memory.is_full:
            return {}
        batch = self.memory.as_batch().to(self.device)
        self.memory.clear()
        return self.train(batch, time_step)

    def update_episode(self, episode: Episode, episode_num: int, time_step: int):
        if not self.memory.update_on_episodes:
            return {}
        self.memory.add(episode)
        if not self.memory.is_full:
            return {}
        batch = self.memory.as_batch().to(self.device)
        self.memory.clear()
        return self.train(batch, time_step)

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
