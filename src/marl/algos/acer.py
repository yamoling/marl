from collections import defaultdict
from copy import deepcopy
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import torch
from marlenv import Episode, Transition

from marl.models import Batch, EpisodeMemory, Mixer, QNetwork, Trainer
from marl.models.batch import EpisodeBatch
from marl.models.nn import CategoricalActor
from marl.utils import Schedule
from marl.utils.tuning import tuning

EPS = 1e-8
"""Numerical stability constant used when dividing by probabilities."""


@dataclass
class ACER(Trainer):
    """
    Actor-Critic with Experience Replay (ACER) for multi-agent, discrete action spaces.

    Paper: https://arxiv.org/abs/1611.01224 (Wang et al., ICLR 2017).

    The trainer implements the discrete part of the paper (Sections 3 and 4):
     - Retrace(λ=1) multi-step estimation of the action-value function;
     - truncated importance sampling with bias correction for the policy gradient;
     - the efficient trust region update based on a running average policy network.

    ## Divergences from the paper
     - The continuous variant (Section 5, stochastic duelling networks) is not implemented.
     - The paper trains 16 asynchronous actor-learners that each own a replay memory of 50,000 frames
       and update every `k = 20` steps. Here a single learner updates on whole trajectories, and
       `memory_size` is expressed in episodes rather than in frames.
     - The Retrace trace is always Retrace(λ=1), as in Equation (5); `retrace_threshold` only controls
       the truncation `c` of the trace (0 recovers a one-step target, +inf an uncorrected Q(λ)).
     - The entropy bonus is added to the loss *after* the trust region projection of Section 3.3,
       whereas the paper projects the gradient of the regularized objective. The realized KL
       divergence with the average policy can therefore slightly exceed `trust_region_delta`.
     - The agent-wise terms of the loss (the policy gradient and the entropy bonus) are normalised by
       the number of agent time steps and the critic loss by the number of joint time steps, so that
       the effective actor learning rate does not depend on the number of agents nor on whether a
       mixer is used. This has no counterpart in the single-agent paper.

    ## Multi-agent extension
    Just like `PPO` is either IPPO (no mixer) or MAPPO (with a mixer), `ACER` is either fully
    decentralised (`mixer=None`) or centralised (with a mixer):

     - **Decentralised (IACER)**: every agent is an independent ACER learner. The team reward is
       broadcast to every agent, the critic utilities `Q_i(o_i, a)` are used as-is and the Retrace
       trace is computed with the agent's own importance weight `ρ_i = π_i(a_i|o_i) / µ_i(a_i|o_i)`.
     - **Centralised (MACER)**: the per-agent utilities are mixed into a joint `Q_tot` with the state
       (as in QMix/VDN), the Retrace recursion is performed on the joint action-value function and the
       trace uses the joint importance weight `ρ = Π_i ρ_i`. The joint state value is approximated by
       mixing the per-agent state values `V_i(o_i) = Σ_a π_i(a|o_i) Q_i(o_i, a)`, which is exact for
       linear mixers (VDN) and an approximation otherwise.

    In both cases, the policy gradient of agent `i` uses its own (truncated) marginal importance
    weight, and the bias correction term is computed on the agent utilities `Q_i` and `V_i` because
    summing over the joint action space is intractable.

    ## Notes
     - ACER learns from trajectories: the `train_interval` must be expressed in episodes.
     - The behaviour policy µ is read from the transitions (key `action_probabilities`), which the
       agent returned by `make_agent()` stores at every time step.
    """

    actor: CategoricalActor
    critic: QNetwork
    """Critic network that outputs the utility Q_i(o_i, a) of every action for every agent."""
    mixer: Mixer | None
    _: KW_ONLY
    train_interval: tuple[int, Literal["step", "episode"]] = (1, "episode")
    memory_size: int = field(default=5_000, metadata=tuning(500, 20_000))
    """Capacity of the replay memory, in episodes."""
    batch_size: int = field(default=8, metadata=tuning(4, 32))
    """Number of episodes sampled from the replay memory at every off-policy update."""
    replay_ratio: float = field(default=4.0, metadata=tuning(0.0, 8.0))
    """Average number of off-policy (replay) updates performed after each on-policy update. The
    actual number of replay updates is sampled from a Poisson distribution, as in the paper."""
    replay_start: int = 32
    """Minimal number of episodes in the replay memory before starting the off-policy updates."""
    lr_actor: float = field(default=5e-4, metadata=tuning(1e-5, 1e-2, log=True))
    lr_critic: float = field(default=1e-3, metadata=tuning(1e-5, 1e-2, log=True))
    truncation_threshold: float = field(default=10.0, metadata=tuning(1.0, 20.0))
    """Importance weight truncation `c` of the policy gradient (Equation (9) of the paper)."""
    retrace_threshold: float = 1.0
    """Importance weight truncation of the Retrace trace, i.e. the `c` of Retrace(λ) (Munos et al., 2016)."""
    critic_coef: float = 0.5
    """Weight of the critic loss in the total loss."""
    entropy_coef: Schedule = field(default_factory=lambda: Schedule.constant(1e-3))
    """Weight of the entropy bonus."""
    trust_region: bool = True
    """Whether to perform the efficient trust region policy update of Section 3.3 of the paper."""
    trust_region_delta: float = field(default=1.0, metadata=tuning(0.1, 2.0))
    """Trust region constraint `δ` on the linearized KL divergence with the average policy."""
    trust_region_decay: float = 0.99
    """Soft update rate `α` of the average policy network."""

    def __post_init__(self):
        super().__post_init__()
        if self.train_interval[1] != "episode":
            raise ValueError("ACER learns from trajectories: the train interval must be in episodes.")
        self.memory = EpisodeMemory(self.memory_size)
        self.avg_actor = deepcopy(self.actor)
        for param in self.avg_actor.parameters():
            param.requires_grad_(False)
        self._episodes = list[Episode]()
        self._parameters = [*self.actor.parameters(), *self.critic.parameters()]
        if self.mixer is not None:
            self._parameters += list(self.mixer.parameters())
        self._optimizer = self._make_optimizer()

    @property
    def name(self):
        if self.mixer is None:
            return "IACER"
        return f"MACER-{self.mixer.name}"

    def _make_optimizer(self, fused: bool = False):
        """
        Build the AdamW optimizer over the actor, critic and mixer parameter groups.

        @ai-generated
        """
        param_groups: list[dict[str, Any]] = [
            {"params": list(self.actor.parameters()), "lr": self.lr_actor, "name": "actor parameters"},
            {"params": list(self.critic.parameters()), "lr": self.lr_critic, "name": "critic parameters"},
        ]
        if self.mixer is not None:
            param_groups.append(
                {"params": list(self.mixer.parameters()), "lr": self.lr_critic, "name": "mixer parameters"}
            )
        return torch.optim.AdamW(param_groups, eps=1e-5, fused=fused)

    def to(self, device: torch.device) -> Self:
        """
        Send the networks to the given device and rebuild the optimizer so that it can use the fused
        implementation on CUDA.

        @ai-generated
        """
        super().to(device)
        self._optimizer = self._make_optimizer(fused=device.type == "cuda")
        return self

    def randomize(self, method: Literal["xavier", "orthogonal"] = "xavier"):
        """
        Randomize the networks and re-synchronise the average policy network with the actor so that the
        trust region constraint is not violated from the very first update.

        @ai-generated
        """
        super().randomize(method)
        self.avg_actor.load_state_dict(self.actor.state_dict())

    def save(self, directory: Path):
        """
        Save the networks. The average policy network is an exact copy of the actor and therefore shares
        its weights file: the actor is saved last so that the persisted weights are the actor's (and the
        average policy is simply re-initialised to the actor when loading).

        @ai-generated
        """
        super().save(directory)
        self.actor.save(directory)

    def make_agent(self):
        from marl.agents import SimpleAgent

        return SimpleAgent(self.actor, record_probabilities=True)

    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        return {}

    def update_episode(self, episode: Episode, episode_num: int, time_step: int) -> dict[str, Any]:
        """
        Store the episode and, every `train_interval` episodes, perform one on-policy update followed by
        a Poisson-distributed number of off-policy (replay) updates, as in Algorithm 1 of the paper.

        @ai-generated
        """
        self.memory.add(episode)
        self._episodes.append(episode)
        if len(self._episodes) < self.train_interval[0]:
            return {}
        self.entropy_coef.update(time_step)
        on_policy_batch = EpisodeBatch(self._episodes).to(self.device)
        self._episodes = []
        logs = defaultdict(list)
        for key, value in self._update(on_policy_batch, time_step, on_policy=True).items():
            logs[key].append(value)
        n_replays = self._n_replay_updates()
        for _ in range(n_replays):
            batch = self.memory.sample(self.batch_size).to(self.device)
            assert isinstance(batch, EpisodeBatch)
            for key, value in self._update(batch, time_step, on_policy=False).items():
                logs[key].append(value)
        return {
            "acer/n-replay-updates": n_replays,
            "acer/entropy-coef": self.entropy_coef.value,
            **{f"acer/{key}": float(np.mean(values)) for key, values in logs.items()},
        }

    def _n_replay_updates(self):
        """
        Number of off-policy updates to perform, sampled from a Poisson distribution of parameter
        `replay_ratio` (Algorithm 1 of the paper). Zero as long as the replay memory is too small.

        @ai-generated
        """
        if self.replay_ratio <= 0 or len(self.memory) < max(self.replay_start, self.batch_size):
            return 0
        return int(np.random.poisson(self.replay_ratio))

    def _update(self, batch: EpisodeBatch, time_step: int, on_policy: bool) -> dict[str, float]:
        """
        Perform a single ACER gradient step on the given batch of (padded) episodes.

        The intrinsic rewards (when an `ir_module` is set) are recomputed at every update since the
        replayed episodes are off-policy, but the intrinsic reward module itself is only trained on the
        on-policy batches to avoid training it several times on the same data.

        @ai-generated
        """
        if self.mixer is None:
            individual_batch = batch.for_individual_learners()
            assert isinstance(individual_batch, EpisodeBatch)
            batch = individual_batch
        ir_logs = dict[str, float]()
        if self.ir_module is not None:
            batch.rewards = batch.rewards + self.ir_module.compute(batch)
            if on_policy:
                ir_logs = self.ir_module.update(batch, time_step)
        actions = batch.actions.unsqueeze(-1)  # (T, B, A, 1)
        masks = batch.masks  # (T, B) with a mixer, (T, B, A) without
        if masks.dim() == batch.actions.dim():
            agent_masks = masks
        else:
            # With a mixer, the masks are joint: expand them over the agents so that the agent-wise
            # terms of the loss are normalised by the number of agent time steps in both cases.
            agent_masks = masks.unsqueeze(-1).expand_as(batch.actions)
        n_items = batch.n_items
        """Number of (joint) time steps in the batch, i.e. the normaliser of the critic loss."""
        n_agent_items = agent_masks.sum()
        """Number of agent time steps in the batch, i.e. the normaliser of the actor and entropy losses."""

        # Behaviour policy µ(.|x) of every agent, as recorded when the episode was collected.
        mu = batch["action_probabilities"]  # (T, B, A, n_actions)

        # Target policy π(.|x) and its Q-values.
        probs = self._probabilities(self.actor, batch)  # (T, B, A, n_actions)
        log_probs = torch.log(probs.clamp_min(EPS))
        qvalues = self._qvalues(batch.obs, batch.extras)  # (T, B, A, n_actions)
        q_taken = qvalues.gather(-1, actions).squeeze(-1)  # (T, B, A)
        # V_i(x) = E_{a ~ π_i}[Q_i(x, a)]. Only used as a (detached) baseline and in the Retrace recursion.
        values = torch.sum(probs * qvalues, dim=-1).detach()  # (T, B, A)

        # Importance weights: ρ(a) for every action, and ρ for the action that was taken.
        all_ratios = probs.detach() / mu.clamp_min(EPS)  # (T, B, A, n_actions)
        ratios = all_ratios.gather(-1, actions).squeeze(-1)  # (T, B, A)

        if self.mixer is not None:
            q_total = self.mixer.forward(q_taken, batch.states, batch.states_extras)  # (T, B)
            with torch.no_grad():
                v_total = self.mixer.forward(values, batch.states, batch.states_extras)  # (T, B)
            trace_ratios = torch.prod(ratios, dim=-1)  # joint importance weight Π_i ρ_i
        else:
            q_total = q_taken
            v_total = values
            trace_ratios = ratios

        with torch.no_grad():
            next_values = self._next_values(batch)
            q_ret = self._retrace(batch, q_total.detach(), v_total, trace_ratios, next_values)
            advantages = q_ret - v_total
            if self.mixer is not None:
                # The joint advantage is shared by all the agents.
                advantages = advantages.unsqueeze(-1)

        # First term of Equation (9): truncated importance sampling.
        truncated_ratios = ratios.clamp(max=self.truncation_threshold)
        objective = truncated_ratios * log_probs.gather(-1, actions).squeeze(-1) * advantages
        # Second term of Equation (9): bias correction over the actions whose weight was truncated.
        with torch.no_grad():
            bias_weights = (1 - self.truncation_threshold / all_ratios.clamp_min(EPS)).clamp(min=0) * probs
            bias_advantages = qvalues.detach() - values.unsqueeze(-1)
        objective = objective + torch.sum(bias_weights * log_probs * bias_advantages, dim=-1)
        objective = torch.sum(objective * agent_masks)

        if self.trust_region:
            actor_loss, kl_divergence, trust_factor = self._trust_region_loss(
                batch, probs, objective, agent_masks, n_agent_items
            )
        else:
            actor_loss = -objective / n_agent_items
            with torch.no_grad():
                kl_divergence = self._kl_divergence(batch, probs, agent_masks)
            trust_factor = 0.0

        entropy = -torch.sum(probs * log_probs, dim=-1)
        entropy_loss = -torch.sum(entropy * agent_masks) / n_agent_items
        td_error = q_ret - q_total
        critic_loss = 0.5 * torch.sum(td_error**2 * masks) / n_items
        loss = actor_loss + self.critic_coef * critic_loss + self.entropy_coef * entropy_loss

        self._optimizer.zero_grad()
        loss.backward()
        logs = {
            **ir_logs,
            "actor-loss": actor_loss.item(),
            "critic-loss": critic_loss.item(),
            "entropy": -entropy_loss.item(),
            "loss": loss.item(),
            "kl-divergence": kl_divergence,
            "trust-factor": trust_factor,
            "mean-importance-weight": torch.sum(ratios * agent_masks).item() / n_agent_items.item(),
            "mean-q-ret": torch.sum(q_ret * masks).item() / n_items.item(),
        }
        if self.grad_norm_clipping is not None:
            norm = torch.nn.utils.clip_grad_norm_(self._parameters, self.grad_norm_clipping)
            logs["grad-norm"] = norm.item()
        self._optimizer.step()
        self._update_average_actor()
        return logs

    def _probabilities(self, actor: CategoricalActor, batch: Batch, next_obs: bool = False):
        """
        Probability of every action for every agent, i.e. the statistics `φ(x)` of the categorical policy.

        The time and episode dimensions are merged into a single batch dimension before the forward pass
        because agent-wise (`independent`) networks only accept a single batch dimension.

        @ai-generated
        """
        if next_obs:
            obs, extras, available = batch.next_obs, batch.next_extras, batch.next_available_actions
        else:
            obs, extras, available = batch.obs, batch.extras, batch.available_actions
        n_steps, n_episodes = obs.shape[:2]
        logits = actor.forward(
            obs.flatten(0, 1),
            extras.flatten(0, 1),
            available_actions=available.flatten(0, 1),
        )
        return torch.softmax(logits, dim=-1).unflatten(0, (n_steps, n_episodes))

    def _qvalues(self, obs: torch.Tensor, extras: torch.Tensor):
        """
        Utilities `Q_i(o_i, a)` of every agent for every action, with the time and episode dimensions
        merged into a single batch dimension for the forward pass (see `_probabilities`).

        @ai-generated
        """
        n_steps, n_episodes = obs.shape[:2]
        qvalues = self.critic.batch_qvalues(obs.flatten(0, 1), extras.flatten(0, 1))
        return qvalues.unflatten(0, (n_steps, n_episodes))

    def _next_values(self, batch: EpisodeBatch) -> torch.Tensor:
        """
        State values `V(x_{t+1})` used to bootstrap the Retrace recursion of truncated episodes.

        @ai-generated
        """
        next_probs = self._probabilities(self.actor, batch, next_obs=True)
        next_qvalues = self._qvalues(batch.next_obs, batch.next_extras)
        next_values = torch.sum(next_probs * next_qvalues, dim=-1)
        if self.mixer is not None:
            next_values = self.mixer.forward(next_values, batch.next_states, batch.next_states_extras)
        return next_values

    def _retrace(
        self,
        batch: EpisodeBatch,
        qvalues: torch.Tensor,
        values: torch.Tensor,
        ratios: torch.Tensor,
        next_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retrace(λ=1) estimate of the action-value function (Equation (5) of the paper), computed
        backwards in time over the (padded) episodes of the batch:

        `Q_ret(x_t, a_t) = r_t + γ [ c_{t+1} (Q_ret(x_{t+1}, a_{t+1}) - Q(x_{t+1}, a_{t+1})) + V(x_{t+1})]`

        where `c_t = min(retrace_threshold, ρ_t)`. Padded time steps are zeroed out so that they cannot
        leak into the recursion, and episodes that were truncated (as opposed to terminated) bootstrap
        on `V(x_{t+1})` at their last time step.

        @ai-generated
        """
        masks = batch.masks
        rewards = batch.rewards
        not_dones = batch.not_dones.to(rewards.dtype)
        traces = ratios.clamp(max=self.retrace_threshold)
        # Index of the last (non-padded) time step of each episode.
        last_index = masks.sum(dim=0).long() - 1
        time_steps = torch.arange(masks.shape[0], device=masks.device).view(-1, *(1 for _ in last_index.shape))
        is_last = time_steps == last_index.unsqueeze(0)

        q_ret = torch.zeros_like(rewards)
        # Correction term of the next time step, i.e. the bracket of the equation above.
        next_correction = torch.zeros_like(rewards[0])
        for t in range(masks.shape[0] - 1, -1, -1):
            # At the end of an episode, either bootstrap on V(x_{t+1}) (truncation) or stop (termination).
            bootstrap = torch.where(is_last[t], not_dones[t] * next_values[t], next_correction)
            current = (rewards[t] + self.gamma * bootstrap) * masks[t]
            q_ret[t] = current
            next_correction = masks[t] * (traces[t] * (current - qvalues[t]) + values[t])
        return q_ret

    def _kl_divergence(self, batch: EpisodeBatch, probs: torch.Tensor, agent_masks: torch.Tensor) -> float:
        """
        Average `D_KL[π_avg(.|x) || π(.|x)]` between the average policy network and the current policy.

        @ai-generated
        """
        avg_probs = self._probabilities(self.avg_actor, batch)
        kl = torch.sum(avg_probs * (torch.log(avg_probs.clamp_min(EPS)) - torch.log(probs.clamp_min(EPS))), dim=-1)
        return (torch.sum(kl * agent_masks) / agent_masks.sum()).item()

    def _trust_region_loss(
        self,
        batch: EpisodeBatch,
        probs: torch.Tensor,
        objective: torch.Tensor,
        agent_masks: torch.Tensor,
        n_agent_items: torch.Tensor,
    ):
        """
        Efficient trust region policy update (Section 3.3 of the paper).

        The ACER gradient `g` is computed with respect to the statistics `φ(x)` of the policy (the
        probability vector of the categorical distribution) and projected on the linearized KL
        constraint with the average policy network:

        `z* = g - max(0, (kᵀg - δ) / ||k||²) k`  where  `k = ∇_φ D_KL[π_avg(.|x) || π(.|x)] = -π_avg / φ`

        The resulting direction is then back-propagated through the network with the surrogate loss
        `-φᵀz*`, so that `∂loss/∂θ = -∂φ/∂θ z*`.

        @ai-generated
        """
        (g,) = torch.autograd.grad(objective, probs, retain_graph=True)
        with torch.no_grad():
            avg_probs = self._probabilities(self.avg_actor, batch)
            kl = torch.sum(avg_probs * (torch.log(avg_probs.clamp_min(EPS)) - torch.log(probs.clamp_min(EPS))), dim=-1)
            k = -avg_probs / probs.clamp_min(EPS)
            k_dot_g = torch.sum(k * g, dim=-1)
            k_dot_k = torch.sum(k * k, dim=-1)
            factor = torch.where(
                k_dot_k > 0,
                ((k_dot_g - self.trust_region_delta) / k_dot_k.clamp_min(EPS)).clamp(min=0),
                torch.zeros_like(k_dot_k),
            )
            z = g - factor.unsqueeze(-1) * k
        loss = -torch.sum(torch.sum(probs * z, dim=-1) * agent_masks) / n_agent_items
        kl_divergence = (torch.sum(kl * agent_masks) / agent_masks.sum()).item()
        trust_factor = (torch.sum(factor * agent_masks) / agent_masks.sum()).item()
        return loss, kl_divergence, trust_factor

    def _update_average_actor(self):
        """
        Soft update of the average policy network: `θ_a <- α θ_a + (1 - α) θ`.

        @ai-generated
        """
        with torch.no_grad():
            for avg_param, param in zip(self.avg_actor.parameters(), self.actor.parameters()):
                avg_param.mul_(self.trust_region_decay).add_(param, alpha=1 - self.trust_region_decay)
