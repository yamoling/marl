from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from marlenv import Transition
from marlenv.utils import Schedule

from marl.agents import SimpleAgent
from marl.models import Mixer
from marl.models.batch import Batch, TransitionBatch
from marl.models.nn import ActorCritic
from marl.models.trainer import Trainer


@dataclass
class PPO(Trainer):
    """
    Proximal Policy Optimization (PPO) training algorithm. If a value mixer is provided, this is MAPPO (Multi-Agent PPO). Otherwise,
    it is IPPO (Independent PPO).
    PPO paper: https://arxiv.org/abs/1707.06347
    MAPPO/IPPO paper: https://arxiv.org/abs/2103.01955
    """

    actor_critic: ActorCritic
    batch_size: int
    c1: Schedule
    c2: Schedule
    eps_clip: float
    gae_lambda: float
    gamma: float
    lr: float
    minibatch_size: int
    n_epochs: int
    value_mixer: Optional[Mixer]
    grad_norm_clipping: Optional[float]

    def __init__(
        self,
        actor_critic: ActorCritic,
        gamma: float,
        value_mixer: Optional[Mixer] = None,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        n_epochs: int = 64,
        eps_clip: float = 0.2,
        critic_c1: Schedule | float = 0.5,
        exploration_c2: Schedule | float = 0.01,
        train_interval: int = 2048,
        minibatch_size: Optional[int] = None,
        gae_lambda: float = 0.95,
        grad_norm_clipping: Optional[float] = None,
    ):
        """
        Parameters
        - `actor_critic`: The actor-critic neural network
        - `value_mixer`: The mixer to use for the value function
        - `gamma`: The discount factor
        - `lr_actor`: The learning rate for the actor
        - `lr_critic`: The learning rate for the critic
        - `n_epochs`: The number of epochs (K) to train the model, i.e. the number of gradient steps
        - `eps_clip`: The clipping parameter for the PPO loss
        - `critic_c1`: The coefficient for the critic loss
        - `exploration_c2`: The coefficient for the entropy loss
        - `train_interval`: The number of steps between training iterations, i.e. the number of steps to collect before training
        - `minibatch_size`: The size of the minibatches to use for training, must be lower or equal to `train_interval`
        - `gae_lambda`: The lambda parameter (trace decay) for the generalized advantage estimation
        - `grad_norm_clipping`: The maximum norm of the gradients at each epoch
        """
        super().__init__("step")
        self.batch_size = train_interval
        if minibatch_size is None:
            minibatch_size = train_interval
        self.minibatch_size = minibatch_size
        self.actor_critic = actor_critic
        self.value_mixer = value_mixer
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.eps_clip = eps_clip
        self._memory = []
        self._ratio_min = 1 - eps_clip
        self._ratio_max = 1 + eps_clip
        param_groups, self._parameters = self._compute_param_groups(lr_actor, lr_critic)
        self.optimizer = torch.optim.Adam(param_groups)
        if isinstance(critic_c1, (float, int)):
            critic_c1 = Schedule.constant(critic_c1)
        self.c1 = critic_c1
        if isinstance(exploration_c2, (float, int)):
            exploration_c2 = Schedule.constant(exploration_c2)
        self.c2 = exploration_c2
        self.gae_lambda = gae_lambda
        self.grad_norm_clipping = grad_norm_clipping

    def _compute_param_groups(self, lr_actor: float, lr_critic: float):
        all_parameters = list(self.actor_critic.parameters())
        params = [
            {"params": self.actor_critic.policy_parameters, "lr": lr_actor, "name": "actor parameters"},
            {"params": self.value_parameters, "lr": lr_critic, "name": "critic parameters"},
            {"params": self.actor_critic.shared_parameters, "lr": min(lr_actor, lr_critic), "name": "shared parameters"},
        ]
        if self.value_mixer is not None:
            params += [{"params": self.value_mixer.parameters(), "lr": lr_critic, "name": "mixer parameters"}]
            all_parameters += list(self.value_mixer.parameters())
        return params, all_parameters

    def _compute_training_data(self, batch: Batch):
        """Compute the returns, advantages and action log_probs according to the current policy"""
        policy = self.actor_critic.policy(batch.obs, batch.extras, batch.available_actions)
        log_probs = policy.log_prob(batch.actions)
        all_values = self.actor_critic.value(batch.all_obs, batch.all_extras)
        if self.value_mixer is not None:
            all_values = self.value_mixer.forward(all_values, batch.states)

        advantages = batch.compute_gae(self.gamma, all_values, self.gae_lambda)
        returns = advantages + all_values[:-1]

        if self.value_mixer is not None:
            # Since we later multiply the ratios (derived from log_probs) by the advantages,
            # we need to expand the advantages in order to have the same shape as the log_probs
            while advantages.dim() < log_probs.dim():
                advantages = advantages.unsqueeze(-1)
            advantages = advantages.expand_as(log_probs)
        return returns, advantages, log_probs

    def train(self, batch: Batch, time_step: int) -> dict[str, float]:
        # batch.normalize_rewards()
        if self.value_mixer is None:
            batch.for_individual_learners()
        self.c1.update(time_step)
        self.c2.update(time_step)
        with torch.no_grad():
            returns, advantages, log_probs = self._compute_training_data(batch)

        min_loss = 0.0
        max_loss = 0.0
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_entropy_loss = 0.0
        total_loss = 0.0
        total_norm = torch.zeros(1, dtype=torch.float32, device=self._device)
        # Perform the minibatch updates
        for _ in range(self.n_epochs):
            indices = np.random.choice(self.batch_size, self.minibatch_size, replace=False)
            minibatch = batch.get_minibatch(indices)
            mini_log_probs, mini_returns, mini_advantages = log_probs[indices], returns[indices], advantages[indices]

            # Use the Monte Carlo estimate of returns as target values
            # L^VF(θ) = E[(V(s) - V_targ(s))^2] in PPO paper
            mini_policy, mini_values = self.actor_critic.forward(minibatch.obs, minibatch.extras, minibatch.available_actions)
            if self.value_mixer is not None:
                mini_values = self.value_mixer.forward(mini_values, minibatch.states)
            critic_loss = torch.nn.functional.mse_loss(mini_values, mini_returns)

            # Actor loss (ratio between the new and old policy):
            # L^CLIP(θ) = E[ min(r(θ)A, clip(r(θ), 1 − ε, 1 + ε)A) ] in PPO paper
            new_log_probs = mini_policy.log_prob(minibatch.actions)
            # assert new_log_probs.shape == mini_log_probs.shape == mini_advantages.shape
            ratios = torch.exp(new_log_probs - mini_log_probs)
            surrogate1 = mini_advantages * ratios
            surrogate2 = torch.clamp(ratios, self._ratio_min, self._ratio_max) * mini_advantages
            # Minus because we want to maximize the objective
            actor_loss = -torch.min(surrogate1, surrogate2).mean()

            # S[\pi_0](s_t) in the paper
            entropy_loss = mini_policy.entropy().mean()

            self.optimizer.zero_grad()
            # Equation (9) in the paper
            loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy_loss
            loss.backward()
            if self.grad_norm_clipping is not None:
                total_norm += torch.nn.utils.clip_grad_norm_(self._parameters, self.grad_norm_clipping)
            self.optimizer.step()

            # Bookkeeping
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_loss += loss.item()
            min_loss = min(min_loss, loss.item())
            max_loss = max(max_loss, loss.item())

        self._memory.clear()
        logs = {
            "avg_actor_loss": total_actor_loss / self.n_epochs,
            "avg_critic_loss": total_critic_loss / self.n_epochs,
            "avg_entropy_loss": total_entropy_loss / self.n_epochs,
            "avg_loss": total_loss / self.n_epochs,
            "min_loss": min_loss,
            "max_loss": max_loss,
        }
        # for i, layer in enumerate(self.actor_critic.value_parameters):
        #     logs[f"critic-layer-{i} mean"] = layer.data.mean().item()
        # for i, layer in enumerate(self.actor_critic.policy_parameters):
        #     logs[f"actor-layer-{i} mean"] = layer.data.mean().item()

        if self.grad_norm_clipping is not None:
            logs["total_grad_norm"] = total_norm.item()
        return logs

    @property
    def device(self):
        return self._device

    @property
    def value_parameters(self):
        params = self.actor_critic.value_parameters
        if self.value_mixer is not None:
            params += self.value_mixer.parameters()
        return params

    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        self._memory.append(transition)
        if len(self._memory) == self.batch_size:
            batch = TransitionBatch(self._memory).to(self._device)
            logs = self.train(batch, time_step)
            return logs
        return {}

    def make_agent(self):
        return SimpleAgent(self.actor_critic)
