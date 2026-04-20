from collections import defaultdict
from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from typing import Literal, cast

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
    - Termination update follows Theorem 2, using
      $A_\Omega(s', \omega) = Q_\Omega(s', \omega) - V_\Omega(s')$ at next state.
    - Intra-option policy is optimized with a dual-clipped PPO surrogate (Ye et al., AAAI 2020) using advantages built from the option-value critic.
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
    dual_clip_c: float = 3.0
    """Lower-bound constant for the dual-clip PPO objective (Ye et al., AAAI 2020).
    For negative-advantage samples, the surrogate is floored at ``dual_clip_c * Â``,
    zeroing the gradient when the ratio exceeds ``dual_clip_c``.  Must be >= 1.
    The original paper uses c = 3."""
    train_interval: int = 64
    minibatch_size: int = 32
    normalize_advantages: bool = True
    grad_norm_clipping: float | None = 0.5
    train_on: Literal["episode", "transition"] = "transition"
    q_updater: InitVar[TargetParametersUpdater | None] = None
    target_updater: TargetParametersUpdater = field(init=False)
    option_train_policy: Policy = field(default_factory=lambda: EpsilonGreedy.constant(0.1))
    optimizer: torch.optim.Optimizer = field(init=False)
    memory: ReplayMemory[Transition | Episode, Batch] = field(init=False)

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
        self._parameters = list(self.oc.parameters())
        if self.mixer is not None:
            self._parameters.extend(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(self._parameters, lr=self.lr)
        if isinstance(critic_c1, (float, int)):
            critic_c1 = Schedule.constant(critic_c1)
        self.c1 = critic_c1
        if isinstance(entropy_c2, (float, int)):
            entropy_c2 = Schedule.constant(entropy_c2)
        self.c2 = entropy_c2

        if self.train_on == "transition":
            memory = TransitionMemory(self.train_interval)
        else:
            memory = EpisodeMemory(self.train_interval)
        self.memory = cast(ReplayMemory[Transition | Episode, Batch], memory)

        if q_updater is None:
            q_updater = HardUpdate(200)
        self.target_updater = q_updater
        self.target_updater.add_parameters(self.oc.parameters(), self.target_oc.parameters())
        if self.mixer is not None and self.target_mixer is not None:
            self.target_updater.add_parameters(self.mixer.parameters(), self.target_mixer.parameters())

    @property
    def n_options(self):
        return self.oc.n_options

    def _compute_training_data(self, batch: Batch):
        options = batch["options"].unsqueeze(-1)
        with torch.no_grad():
            # Values computations
            q_options = self.target_oc.compute_q_options(batch.obs, batch.extras)
            values = torch.gather(q_options, dim=-1, index=options).squeeze(-1)
            if self.target_mixer is not None:
                values = self.target_mixer.forward(values, batch.states)
            # Next values computations
            next_values = self.target_oc.value_on_arrival(batch.next_obs, batch.next_extras, options)
            if self.target_mixer is not None:
                next_values = self.target_mixer.forward(next_values, batch.next_states)
        values[batch.masked_indices] = 0.0
        next_values[batch.dones] = 0.0
        advantages = batch.compute_gae(self.gamma, values, next_values, trace_decay=self.gae_lambda, normalize=self.normalize_advantages)
        advantages[batch.masked_indices] = 0.0
        returns = advantages + values
        return returns, advantages

    def train(self, batch: Batch, step_num: int) -> dict[str, float]:
        self.c1.update(step_num)
        self.c2.update(step_num)

        if self.mixer is None:
            batch = batch.for_individual_learners()

        with torch.no_grad():
            dist = self.oc.policy(batch.obs, batch.extras, batch.available_actions, batch["options"].unsqueeze(-1))
            old_log_probs = dist.log_prob(batch.actions)
            returns, advantages = self._compute_training_data(batch)
            if self.mixer is not None:
                # We need agent-wise advantages, but if there is a mixer, the advantages are computed on the joint value so
                # we need to repeat them for each agent.
                advantages = advantages.repeat_interleave(batch.n_agents).view(*advantages.shape, batch.n_agents)

        log_lists = defaultdict(list)
        for _ in range(self.n_epochs):
            # Minibatch setup
            minibatch, indices = self._minibatch_setup(batch)
            mini_options = minibatch["options"].unsqueeze(-1)
            mini_returns = returns[*indices]
            mini_actor_advantages = advantages[*indices, :]
            mini_old_log_probs = old_log_probs[indices]

            critic_loss = self._compute_critic_loss(minibatch, mini_returns, mini_options)
            actor_loss, entropy_loss, extra_logs = self._compute_actor_loss(
                minibatch,
                mini_options,
                mini_actor_advantages,
                mini_old_log_probs,
            )
            termination_loss = self._compute_termination_loss(minibatch, mini_options)

            self.optimizer.zero_grad()
            loss = actor_loss + self.c1 * critic_loss + self.c2 * entropy_loss + self.termination_c3 * termination_loss
            loss.backward()
            if self.grad_norm_clipping is not None:
                norm = torch.nn.utils.clip_grad_norm_(self._parameters, self.grad_norm_clipping)
                log_lists["norms"].append(norm.detach().cpu().item())
            self.optimizer.step()

            log_lists["actor_loss"].append(actor_loss.item())
            log_lists["critic_loss"].append(critic_loss.item())
            log_lists["entropy_loss"].append(entropy_loss.item())
            log_lists["termination_loss"].append(termination_loss.item())
            log_lists["loss"].append(loss.item())
            for key, values in extra_logs.items():
                log_lists[key].append(values)

        logs = {
            "ppoc/c1": self.c1.value,
            "ppoc/c2": self.c2.value,
            **self.option_train_policy.update(step_num),
            **self.target_updater.update(step_num),
            **{f"ppoc/mean-{key}": np.mean(values) for key, values in log_lists.items()},
            **{f"ppoc/max-{key}": np.max(values) for key, values in log_lists.items()},
            **{f"ppoc/min-{key}": np.min(values) for key, values in log_lists.items()},
        }
        return logs

    def _minibatch_setup(self, batch: Batch):
        # Minibatch setup
        indices = np.random.choice(batch.size, self.minibatch_size, replace=False)
        minibatch = batch.get_minibatch(indices).to(self.device)
        if self.mixer is None:
            minibatch = minibatch.for_individual_learners()
        if isinstance(minibatch, EpisodeBatch):
            indices = (slice(None), indices)
        else:
            indices = (indices,)
        return minibatch, indices

    def _compute_critic_loss(self, minibatch: Batch, mini_returns: torch.Tensor, mini_options: torch.Tensor):
        q_options = self.oc.compute_q_options(minibatch.obs, minibatch.extras)
        mini_values = torch.gather(q_options, dim=-1, index=mini_options).squeeze(-1)
        if self.target_mixer is not None:
            mini_values = self.target_mixer.forward(mini_values, minibatch.states)
        td_error = (mini_values - mini_returns) * minibatch.masks
        critic_loss = torch.sum(td_error**2) / minibatch.masks_sum
        return critic_loss

    def _compute_actor_loss(
        self,
        minibatch: Batch,
        mini_options: torch.Tensor,
        mini_advantages: torch.Tensor,
        mini_old_log_probs: torch.Tensor,
    ):
        mini_dist = self.oc.policy(minibatch.obs, minibatch.extras, minibatch.available_actions, mini_options)
        mini_log_probs = mini_dist.log_prob(minibatch.actions)
        mini_entropy = mini_dist.entropy()

        ratio = torch.exp(mini_log_probs - mini_old_log_probs)
        surrogate1 = ratio * mini_advantages
        surrogate2 = torch.clamp(ratio, self._ratio_min, self._ratio_max) * mini_advantages
        ppo_min = torch.min(surrogate1, surrogate2)

        # Dual-clip PPO (Ye et al., AAAI 2020, arXiv:1912.09729).
        #
        # Standard PPO uses L = min(r·Â, clip(r, 1-ε, 1+ε)·Â).  The clip correctly
        # zeroes the gradient when r > 1+ε and Â > 0 (Case A: policy already moved
        # in the right direction), and when r < 1-ε and Â < 0 (Case C: already
        # corrected enough).  However, when Â < 0 AND r > 1+ε (Case D), torch.min
        # picks the unclipped surrogate r·Â, giving a gradient ∝ r·|Â| that is
        # completely unbounded and drives exponential ratio escalation across epochs.
        #
        # The dual-clip floors the objective at dual_clip_c·Â for negative-advantage
        # samples.  When r > dual_clip_c, ppo_min = r·Â < dual_clip_c·Â (both negative),
        # so torch.max selects the floor, and since dual_clip_c·Â is a constant w.r.t.
        # log π, the gradient is zero — Case D is neutralised.
        dual_clip_floor = self.dual_clip_c * mini_advantages
        actor_objective = torch.where(mini_advantages < 0, torch.max(ppo_min, dual_clip_floor), ppo_min)

        actor_loss = -torch.sum(actor_objective) / minibatch.masks_sum
        entropy_loss = -torch.sum(mini_entropy) / minibatch.masks_sum
        return actor_loss, entropy_loss, {"ratios": ratio.detach().cpu().numpy(), "entropies": mini_entropy.detach().cpu().numpy()}

    def _compute_termination_loss(self, minibatch: Batch, mini_options: torch.Tensor):
        next_termination_probs = self.oc.termination_probability(minibatch.next_obs, minibatch.next_extras, mini_options)
        with torch.no_grad():
            next_q_options = self.target_oc.compute_q_options(minibatch.next_obs, minibatch.next_extras)
            next_q_max = next_q_options.max(dim=-1).values
            next_q_current = torch.gather(next_q_options, dim=-1, index=mini_options).squeeze(-1)
            if self.target_mixer is not None:
                next_q_max = self.target_mixer.forward(next_q_max, minibatch.next_states)
                next_q_current = self.target_mixer.forward(next_q_current, minibatch.next_states)
            next_advantage = next_q_current - next_q_max
            if self.target_mixer is not None:
                next_advantage = next_advantage.repeat_interleave(self.n_agents).view(minibatch.size, self.n_agents)

        termination_mask = minibatch.not_dones.repeat_interleave(self.n_agents).view(minibatch.size, self.n_agents)
        termination_loss = torch.sum(next_termination_probs * (next_advantage + self.termination_reg) * termination_mask)
        return termination_loss

    def update_step(self, transition: Transition, time_step: int) -> dict[str, float]:
        if not self.memory.update_on_transitions:
            return {}
        self.memory.add(transition)
        logs = self.option_train_policy.update(time_step)
        if not self.memory.is_full:
            return logs
        batch = self.memory.as_batch().to(self.device)
        self.memory.clear()
        logs = logs | self.train(batch, time_step)
        return logs

    def update_episode(self, episode: Episode, episode_num: int, time_step: int) -> dict[str, float]:
        if not self.memory.update_on_episodes:
            return {}
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
