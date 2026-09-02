"""
Social influence as intrinsic motivation (Jaques et al., ICML 2019).

Paper: https://proceedings.mlr.press/v97/jaques19a.html

Agents are rewarded for having a causal influence on the actions of the other agents. The influence
of agent `k` on agent `j` at time `t` is the KL divergence between the distribution over `j`'s next
action conditioned on `k`'s actual action, and the same distribution marginalised over the
counterfactual actions that `k` could have taken:

    c_t^k = sum_{j != k} D_KL[ p(a_{t+1}^j | a_t^k, a_t^{-k}, o_t^k) || sum_a' pi^k(a'|o_t^k) p(a_{t+1}^j | a', a_t^{-k}, o_t^k) ]

Following Section 6 of the paper, `p` is estimated with a per-agent *Model of Other Agents* (MOA),
a recurrent network trained by supervised learning to predict the other agents' next actions. This
makes the reward fully decentralised: no agent ever accesses another agent's policy or reward.

This is implemented as a `Trainer` (subclassing `PPO`) rather than as a standalone `IRModule`
because the counterfactual marginalisation requires the *current* policy `pi^k`. An `IRModule` would
have to hold a reference to the actor, which cannot survive (de)serialisation without silently
going stale, whereas a trainer already owns its actor.
"""

from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Literal

import torch
import torch.nn.functional as F
from marlenv import MARLEnv

from marl.algos.ppo import PPO
from marl.env import EnvConfig
from marl.models import NN, Batch, Mixer
from marl.models.batch import EpisodeBatch
from marl.models.nn import Actor, Critic
from marl.nn.model_bank import CNN, MLP
from marl.utils import Schedule


@dataclass
class ModelOfOtherAgents(NN):
    """
    Model of Other Agents (MOA): predicts the next action of every *other* agent from an agent's own
    observation and from the joint action taken at the current time step.

    The network is shared by all agents (parameter sharing). To keep the network agnostic to which
    agent it is modelling, the joint action vector fed to the model is rolled so that the observing
    agent's own action always comes first, and the predictions are ordered accordingly: prediction
    `i` of agent `k` concerns agent `(k + i + 1) % n_agents`.

    @ai-generated
    """

    obs_shape: tuple[int, ...]
    extras_size: int
    n_agents: int
    n_actions: int
    _: KW_ONLY
    hidden_size: int = 128
    output_shape: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        """
        Build the observation encoder, the recurrent core and the prediction head.

        @ai-generated
        """
        self.obs_shape = tuple(self.obs_shape)
        self.output_shape = ((self.n_agents - 1) * self.n_actions,)
        super().__post_init__()
        match self.obs_shape:
            case (size,):
                self.encoder = MLP((self.hidden_size,), size, self.extras_size)
            case (_, _, _) as dimensions:
                self.encoder = _ConvEncoder(dimensions, self.extras_size, self.hidden_size)
            case other:
                raise ValueError(f"Unsupported observation shape: {other}")
        self.input_layer = torch.nn.Linear(self.hidden_size + self.n_agents * self.n_actions, self.hidden_size)
        self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size)
        self.output_layer = torch.nn.Linear(self.hidden_size, self.output_size)

    @property
    def order(self) -> torch.Tensor:
        """
        `order[k, i]` is the index of the agent occupying slot `i` of agent `k`'s (rolled) action
        vector, i.e. `(k + i) % n_agents`. Slot 0 is always the observing agent itself.

        @ai-generated
        """
        agents = torch.arange(self.n_agents, device=self.device)
        return (agents.unsqueeze(1) + agents.unsqueeze(0)) % self.n_agents

    def rolled_actions(self, one_hot_actions: torch.Tensor) -> torch.Tensor:
        """
        Turn joint one-hot actions of shape (T, B, n_agents, n_actions) into per-agent, self-first
        action vectors of shape (T, B, n_agents, n_agents * n_actions).

        @ai-generated
        """
        rolled = one_hot_actions[:, :, self.order]
        return rolled.flatten(start_dim=-2)

    def visible_others(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Boolean mask of shape (T, B, n_agents, n_agents - 1), `True` where the influencee appears in
        the influencer's observation, in the same (rolled) order as the model's outputs.

        This assumes the convention followed by LLE (and by the environments of the paper) that the
        first `n_agents` channels of an observation are the position layers of each agent. It is
        therefore only available for spatial (channel, height, width) observations; flat
        observations have no general channel layout to read.

        @ai-generated
        """
        if len(self.obs_shape) != 3:
            raise ValueError(
                "Visibility can only be derived from (channel, height, width) observations. "
                "Use a spatial obs_type (e.g. 'layered' or 'partial5x5' in LLE), or set visibility='all'."
            )
        agent_layers = obs[..., : self.n_agents, :, :].flatten(start_dim=-2)
        present = agent_layers.ne(0.0).any(dim=-1)  # (T, B, n_agents observer, n_agents target)
        return present.gather(-1, self.order[:, 1:].expand(*present.shape[:-1], self.n_agents - 1))

    def _encode(self, obs: torch.Tensor, extras: torch.Tensor, joint_actions: torch.Tensor):
        """
        Concatenate the encoded observation with the (already rolled) joint action vector.

        @ai-generated
        """
        features = self.encoder.forward(obs, extras)
        return F.relu(self.input_layer.forward(torch.cat((features, joint_actions), dim=-1)))

    def forward_with_history(self, obs: torch.Tensor, extras: torch.Tensor, joint_actions: torch.Tensor):
        """
        Run the MOA over a whole trajectory.

        Args:
            obs: observations of shape (T, B, n_agents, *obs_shape).
            extras: extras of shape (T, B, n_agents, extras_size).
            joint_actions: rolled one-hot joint actions of shape (T, B, n_agents, n_agents * n_actions).

        Returns:
            The logits over the other agents' *next* actions, of shape
            (T, B, n_agents, n_agents - 1, n_actions), and the GRU hidden state *preceding* each
            time step, of shape (T, B, n_agents, hidden_size), which is required to replay
            counterfactuals from the factual history.

        @ai-generated
        """
        time, batch, n_agents = obs.shape[:3]
        inputs = self._encode(obs, extras, joint_actions).reshape(time, batch * n_agents, self.hidden_size)
        hidden, _ = self.gru.forward(inputs)
        previous_hidden = torch.cat((torch.zeros_like(hidden[:1]), hidden[:-1]), dim=0)
        logits = self.output_layer.forward(hidden)
        logits = logits.view(time, batch, n_agents, self.n_agents - 1, self.n_actions)
        return logits, previous_hidden.view(time, batch, n_agents, self.hidden_size)

    def counterfactuals(
        self,
        obs: torch.Tensor,
        extras: torch.Tensor,
        joint_actions: torch.Tensor,
        previous_hidden: torch.Tensor,
    ):
        """
        Predict the other agents' next actions for every action the observing agent could have taken,
        replaying a single GRU step from the factual history.

        Returns:
            Logits of shape (n_actions, T, B, n_agents, n_agents - 1, n_actions) where the leading
            dimension indexes the counterfactual action of the observing agent.

        @ai-generated
        """
        time, batch, n_agents = obs.shape[:3]
        # Overwrite slot 0 (the observing agent's own action) with each candidate action in turn.
        candidates = joint_actions.unsqueeze(0).repeat(self.n_actions, 1, 1, 1, 1)
        candidates[..., : self.n_actions] = torch.eye(self.n_actions, device=self.device).view(
            self.n_actions, 1, 1, 1, self.n_actions
        )
        obs = obs.unsqueeze(0).expand(self.n_actions, *obs.shape)
        extras = extras.unsqueeze(0).expand(self.n_actions, *extras.shape)
        inputs = self._encode(obs, extras, candidates).reshape(1, -1, self.hidden_size)
        hidden = previous_hidden.unsqueeze(0).expand(self.n_actions, *previous_hidden.shape)
        outputs, _ = self.gru.forward(inputs, hidden.reshape(1, -1, self.hidden_size).contiguous())
        logits = self.output_layer.forward(outputs.squeeze(0))
        return logits.view(self.n_actions, time, batch, n_agents, self.n_agents - 1, self.n_actions)

    def __hash__(self):
        return id(self)

    @classmethod
    def from_env(cls, env: MARLEnv[Any] | EnvConfig, hidden_size: int = 128):
        return cls(env.observation_shape, env.extras_shape[0], env.n_agents, env.n_actions, hidden_size=hidden_size)


@dataclass
class _ConvEncoder(NN):
    """
    Convolutional observation encoder that also consumes the `extras` vector.

    @ai-generated
    """

    input_shape: tuple[int, int, int]
    extras_size: int
    hidden_size: int
    output_shape: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        self.output_shape = (self.hidden_size,)
        super().__post_init__()
        self.cnn = CNN(self.input_shape)
        self.mlp = MLP((self.hidden_size,), self.cnn.output_size, self.extras_size, hidden_sizes=(256,))

    def forward(self, obs: torch.Tensor, extras: torch.Tensor):
        """
        Encode the observations into a `hidden_size`-dimensional vector.

        @ai-generated
        """
        return self.mlp.forward(self.cnn.forward(obs), extras)

    def __hash__(self):
        return id(self)


@dataclass
class SocialInfluence(PPO):
    """
    PPO (IPPO or MAPPO) augmented with the social influence intrinsic reward of Jaques et al. (2019),
    computed with a decentralised Model of Other Agents.

    Paper: https://proceedings.mlr.press/v97/jaques19a.html

    @ai-generated
    """

    _: KW_ONLY
    moa: ModelOfOtherAgents
    influence_weight: Schedule = field(default_factory=lambda: Schedule.constant(0.1))
    """Weight (beta) of the influence reward relatively to the extrinsic reward."""
    influence_reward_clip: float | None = 10.0
    """Symmetric clipping of the per-step influence reward. `None` disables clipping."""
    moa_lr: float = 3e-4
    moa_loss_weight: float = 1.0
    moa_updates: int = 1
    """Number of supervised MOA updates per PPO training batch."""
    visibility: Literal["all", "agent-channels"] = "all"
    """
    Which (influencer, influencee) pairs are taken into account, both in the influence reward and in
    the MOA loss. The paper only rewards influence on agents that are inside the influencer's field
    of view, since the MOA predictions are unreliable otherwise.

    - `"all"`: every pair, which is what the paper's restriction amounts to under full
      observability (e.g. LLE with `obs_type="layered"` or `"flattened"`).
    - `"agent-channels"`: read the influencee's presence from the first `n_agents` channels of the
      influencer's observation, as in `ModelOfOtherAgents.visible_others`.
    """

    def __post_init__(self):
        """
        Initialise PPO and the optimiser of the Model of Other Agents.

        @ai-generated
        """
        super().__post_init__()
        if self.moa.n_agents < 2:
            raise ValueError("The social influence reward requires at least two agents")
        if self.moa_updates < 1:
            raise ValueError("moa_updates must be positive")
        self._moa_optimizer = torch.optim.Adam(self.moa.parameters(), lr=self.moa_lr)

    @property
    def name(self):
        return f"SocialInfluence-{super().name}"

    @staticmethod
    def _time_major(batch: Batch, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reshape a batch tensor to the (time, batch, n_agents, ...) layout used by the MOA.

        `EpisodeBatch` tensors already use it. `TransitionBatch` tensors of shape (n, ...) are
        turned into a single sequence of length `n`, since the transitions stored by PPO are
        adjacent in time -- the assumption already made by `Batch.compute_gae`.

        @ai-generated
        """
        if isinstance(batch, EpisodeBatch):
            return tensor
        return tensor.unsqueeze(1)

    def _step_masks(self, batch: Batch) -> torch.Tensor:
        """
        Per (time, batch, agent) validity mask: `False` on padded steps and on the last step of an
        episode, where the "next action" that the MOA predicts belongs to another episode.

        @ai-generated
        """
        masks = self._time_major(batch, batch.masks) * self._time_major(batch, batch.not_dones)
        if masks.dim() == 2:  # (time, batch): the same mask applies to every agent
            masks = masks.unsqueeze(-1).expand(-1, -1, self.moa.n_agents)
        return masks.float()

    def _pair_masks(self, obs: torch.Tensor) -> torch.Tensor | None:
        """
        Per (time, batch, influencer, influencee) mask, or `None` when every pair counts.

        @ai-generated
        """
        if self.visibility == "all":
            return None
        return self.moa.visible_others(obs).float()

    def _moa_inputs(self, batch: Batch):
        """
        Extract the (time, batch, n_agents, ...) tensors that the MOA consumes.

        @ai-generated
        """
        obs = self._time_major(batch, batch.obs)
        extras = self._time_major(batch, batch.extras)
        actions = self._time_major(batch, batch.actions).squeeze(-1)
        one_hot = F.one_hot(actions, self.moa.n_actions).float()
        return obs, extras, actions, self.moa.rolled_actions(one_hot)

    def _influence_reward(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the per-agent social influence reward of shape (T, B, n_agents) along with the raw
        (unweighted, unclipped) influence for logging.

        @ai-generated
        """
        obs, extras, actions, joint_actions = self._moa_inputs(batch)
        with torch.no_grad():
            _, previous_hidden = self.moa.forward_with_history(obs, extras, joint_actions)
            # (n_actions, T, B, n_agents, n_agents - 1, n_actions)
            counterfactual_logits = self.moa.counterfactuals(obs, extras, joint_actions, previous_hidden)
            counterfactual_probs = torch.softmax(counterfactual_logits, dim=-1)
            # p(a_{t+1}^j | a_t^k, ...): select the slice of the action that was actually taken.
            index = actions.view(1, *actions.shape, 1, 1).expand(1, *counterfactual_probs.shape[1:])
            factual = torch.gather(counterfactual_probs, 0, index).squeeze(0)
            # Marginal policy: sum over the counterfactual actions weighted by the agent's own policy.
            # The actor is fed the raw batch tensors: some actors (independent ones) only accept
            # the batch's native number of dimensions.
            policy = self.actor.policy(batch.obs, batch.extras, available_actions=batch.available_actions)
            own_probs = self._time_major(batch, policy.probs).movedim(-1, 0)  # type: ignore[union-attr]
            marginal = torch.sum(own_probs.view(*own_probs.shape, 1, 1) * counterfactual_probs, dim=0)
            # KL(factual || marginal), summed over the influencees.
            eps = torch.finfo(factual.dtype).tiny
            kl = factual * (torch.log(factual + eps) - torch.log(marginal + eps))
            per_influencee = kl.sum(dim=-1)
            pair_masks = self._pair_masks(obs)
            if pair_masks is not None:
                per_influencee = per_influencee * pair_masks
            influence = per_influencee.sum(dim=-1)
            influence = torch.nan_to_num(influence, nan=0.0, posinf=0.0, neginf=0.0)
            reward = influence
            if self.influence_reward_clip is not None:
                reward = reward.clamp(-self.influence_reward_clip, self.influence_reward_clip)
            reward = reward * self.influence_weight
        return reward, influence

    def _update_moa(self, batch: Batch) -> float:
        """
        Train the Model of Other Agents to predict the other agents' next actions with a masked
        cross-entropy loss.

        @ai-generated
        """
        obs, extras, actions, joint_actions = self._moa_inputs(batch)
        # Targets: the action taken at t + 1 by every other agent, in the same (rolled) order as the
        # MOA outputs.
        targets = actions[:, :, self.moa.order[:, 1:]][1:]
        # The prediction made at t is only supervised if t and t + 1 belong to the same episode,
        # and (optionally) only for the influencees that the influencer could actually see.
        masks = self._step_masks(batch)[:-1].unsqueeze(-1).expand_as(targets)
        pair_masks = self._pair_masks(obs)
        if pair_masks is not None:
            masks = masks * pair_masks[:-1]
        total = 0.0
        for _ in range(self.moa_updates):
            logits, _ = self.moa.forward_with_history(obs, extras, joint_actions)
            logits = logits[:-1]
            cross_entropy = F.cross_entropy(
                logits.reshape(-1, self.moa.n_actions),
                targets.reshape(-1),
                reduction="none",
            ).view_as(targets)
            loss = self.moa_loss_weight * torch.sum(cross_entropy * masks) / masks.sum().clamp(min=1)
            self._moa_optimizer.zero_grad()
            loss.backward()
            if self.grad_norm_clipping is not None:
                torch.nn.utils.clip_grad_norm_(self.moa.parameters(), self.grad_norm_clipping)
            self._moa_optimizer.step()
            total += loss.item()
        return total / self.moa_updates

    def add_intrinsic_rewards(self, batch: Batch, time_step: int) -> dict[str, Any]:
        """
        Add the social influence reward to the batch and train the Model of Other Agents.

        @ai-generated
        """
        if batch.reward_size > 1:
            raise ValueError("The social influence reward only supports scalar rewards")
        self.influence_weight.update(time_step)
        reward, influence = self._influence_reward(batch)
        reward = reward * self._step_masks(batch)
        if not isinstance(batch, EpisodeBatch):
            reward = reward.squeeze(1)
        if self.mixer is not None:
            # MAPPO shares a single team reward: aggregate the individual influence rewards.
            reward = reward.sum(dim=-1)
        batch.rewards = batch.rewards + reward
        moa_loss = self._update_moa(batch)
        return {
            "moa-loss": moa_loss,
            "influence-weight": self.influence_weight.value,
            "mean-influence": influence.mean().item(),
            "max-influence": influence.max().item(),
        }

    @classmethod
    def from_env(
        cls,
        env: MARLEnv[Any] | EnvConfig,
        actor: Actor,
        critic: Critic,
        mixer: Mixer | None = None,
        *,
        moa_hidden_size: int = 128,
        **kwargs,
    ):
        """
        Build a `SocialInfluence` trainer with a Model of Other Agents sized for `env`.

        @ai-generated
        """
        moa = ModelOfOtherAgents.from_env(env, hidden_size=moa_hidden_size)
        return cls(actor, critic, mixer, moa=moa, **kwargs)
