from dataclasses import KW_ONLY, dataclass, field
from typing import Any

import torch
from marlenv import MARLEnv

from marl.env import EnvConfig
from marl.models import NN, Batch, IRModule
from marl.nn import model_bank
from marl.utils import Schedule


@dataclass
class ICM(IRModule):
    """
    Intrinsic Curiosity Module (ICM) for discrete action spaces.

    Paper: https://arxiv.org/pdf/1705.05363
    Reference (TensorFlow) implementation: https://github.com/pathak22/noreward-rl

    The module learns a feature space $\\phi$ in which only what the agent can influence is represented,
    and rewards the agent for transitions whose outcome it fails to predict in that space. Three networks
    are trained jointly:

    - the *feature encoder* $\\phi(o)$, provided by the caller;
    - the *inverse model* $\\hat{a}_t = g(\\phi(o_t), \\phi(o_{t+1}))$ (equation 2), whose sole purpose is to
      shape $\\phi$ so that it encodes what is relevant to the agent's own actions and nothing else;
    - the *forward model* $\\hat{\\phi}(o_{t+1}) = f(\\phi(o_t), a_t)$ (equation 4), whose prediction error is
      the curiosity signal.

    The intrinsic reward of equation (6) is
    $$r^i_t = \\frac{\\eta}{2} \\lVert \\hat{\\phi}(o_{t+1}) - \\phi(o_{t+1}) \\rVert^2_2$$
    and the self-supervised loss of equation (7) is $(1 - \\beta) L_I + \\beta L_F$.

    **Multi-agent setting.** The paper is single-agent, so curiosity is computed *per agent*: each agent is
    encoded from its own observation $o^k_t$ and the inverse model predicts its own action $a^k_t$. The
    three networks are shared by all the agents (parameter sharing), which means that an agent is curious
    about transitions that are surprising for the *population*, not just for itself. `compute` therefore
    returns one intrinsic reward per agent, of shape `(*dims, n_agents)`. Trainers that mix the rewards
    into a single team reward (i.e. any trainer with a `mixer`) leave `Batch.individual_rewards` to `False`,
    in which case the per-agent rewards are averaged into a single team-level signal.

    **Multi-objective setting.** Curiosity is a single exploration signal that is not attached to any
    objective in particular: the same intrinsic reward is broadcast over the objectives by the trainer.

    Note:
        Following equation (7) and the reference implementation, the forward loss is *not* detached from
        the encoder: both losses train $\\phi$. The inverse loss is what prevents the encoder from
        collapsing to a trivially predictable (e.g. constant) representation, hence the small default
        `beta`.
    """

    feature_encoder: NN
    """Feature encoder $o \\mapsto \\phi(o)$. Must output `n_features` features and must not be recurrent."""
    n_agents: int
    n_actions: int
    _: KW_ONLY
    n_features: int = 256
    """Size of the feature space in which the forward model predicts, i.e. the output size of the encoder."""
    hidden_size: int = 256
    """Size of the hidden layer of both the inverse and the forward models."""
    beta: float = 0.2
    """Weight of the forward loss against the inverse loss in equation (7). `FORWARD_LOSS_WT` in the reference implementation."""
    lr: float = 1e-3
    """Learning rate of the ICM. Ten times the policy learning rate in the reference implementation (`PREDICTION_LR_SCALE`)."""
    grad_norm_clipping: float | None = 40.0
    weight: Schedule = field(default_factory=lambda: Schedule.constant(0.01))
    """Scaling factor $\\eta$ of equation (6). `PREDICTION_BETA` in the reference implementation."""
    output_shape: tuple[int, ...] = field(init=False)

    def __init__(
        self,
        feature_encoder: NN,
        n_agents: int,
        n_actions: int,
        *,
        n_features: int = 256,
        hidden_size: int = 256,
        beta: float = 0.2,
        lr: float = 1e-3,
        grad_norm_clipping: float | None = 40.0,
        weight: Schedule | None = None,
    ):
        """
        The dataclass-generated `__init__` can not be used here because it would assign the encoder
        submodule before `torch.nn.Module.__init__` has run, which `torch.nn.Module.__setattr__` refuses.
        The encoder is therefore placed in `__dict__` (where `Serializable.to_dict` looks for it) and
        registered as a submodule under a different name by `__post_init__`.

        @ai-generated
        """
        object.__setattr__(self, "feature_encoder", feature_encoder)
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.beta = beta
        self.lr = lr
        self.grad_norm_clipping = grad_norm_clipping
        self.weight = Schedule.constant(0.01) if weight is None else weight
        self.__post_init__()

    def __post_init__(self):
        """Build the inverse and forward models on top of the (already built) encoder. @ai-generated"""
        self.output_shape = (self.n_features,)
        # ⚠️ self.output_shape must be set BEFORE calling super().__post_init__() !
        super().__post_init__()
        if self.feature_encoder.output_size != self.n_features:
            raise ValueError(
                f"The feature encoder outputs {self.feature_encoder.output_size} features but the ICM expects {self.n_features}."
            )
        if self.feature_encoder.is_recurrent:
            raise ValueError("ICM does not support recurrent feature encoders.")
        self.add_module("_feature_encoder", self.feature_encoder)

        # Inverse model (equation 2): φ(o), φ(o') → logits over the actions of the agent.
        self.inverse_model = torch.nn.Sequential(
            torch.nn.Linear(2 * self.n_features, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.n_actions),
        )
        # Forward model (equation 4): φ(o), a → φ(o').
        self.forward_model = torch.nn.Sequential(
            torch.nn.Linear(self.n_features + self.n_actions, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.n_features),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def encode(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the agent-wise observations of the batch into `φ(o_t)` and `φ(o_{t+1})`.

        Both tensors have shape `(*dims, n_agents, n_features)`.

        @ai-generated
        """
        features = self.feature_encoder.forward(batch.obs, batch.extras)
        next_features = self.feature_encoder.forward(batch.next_obs, batch.next_extras)
        return features, next_features

    def forward(self, features: torch.Tensor, next_features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Prediction error of the forward model, i.e. `0.5 * ||φ̂(o') - φ(o')||²₂` of shape `(*dims, n_agents)`.

        This is the intrinsic reward of equation (6) up to the `weight` (η) factor.

        @ai-generated
        """
        one_hot_actions = torch.nn.functional.one_hot(actions, self.n_actions).to(features.dtype)
        predicted = self.forward_model.forward(torch.cat((features, one_hot_actions), dim=-1))
        return 0.5 * (predicted - next_features).square().sum(-1)

    def compute(self, batch: Batch) -> torch.Tensor:
        """
        Curiosity of equation (6), zeroed on the padded time steps.

        Returns a tensor of shape `(*dims, n_agents)` when the batch rewards are agent-wise, and of shape
        `(*dims,)` (the average over the agents) when they are team-wise.

        @ai-generated
        """
        with torch.no_grad():
            features, next_features = self.encode(batch)
            actions = self._agent_actions(batch, features)
            intrinsic_reward = self.weight.value * self.forward(features, next_features, actions)
            intrinsic_reward = intrinsic_reward * self._time_step_masks(batch, features)
            if not batch.individual_rewards:
                intrinsic_reward = intrinsic_reward.mean(-1)
        return intrinsic_reward

    def update(self, batch: Batch, time_step: int) -> dict[str, float]:
        """Minimise `(1 - β) L_I + β L_F` (equation 7) over the padded time steps. @ai-generated"""
        self.weight.update(time_step)
        features, next_features = self.encode(batch)
        actions = self._agent_actions(batch, features)
        masks = self._time_step_masks(batch, features)
        n_items = (masks.sum() * self.n_agents).clamp_min(1)

        # Inverse model loss (equation 3): softmax loss over the action actually taken by each agent.
        logits = self.inverse_model.forward(torch.cat((features, next_features), dim=-1))
        inverse_errors = torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.n_actions),
            actions.reshape(-1),
            reduction="none",
        ).view_as(actions)
        inverse_loss = (inverse_errors * masks).sum() / n_items

        # Forward model loss (equation 5).
        forward_loss = (self.forward(features, next_features, actions) * masks).sum() / n_items

        loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss
        self.optimizer.zero_grad()
        loss.backward()
        logs = {
            "ir-loss": loss.item(),
            "ir-weight": self.weight.value,
            "icm-inverse-loss": inverse_loss.item(),
            "icm-forward-loss": forward_loss.item(),
            "icm-inverse-accuracy": (((logits.argmax(-1) == actions) * masks).sum() / n_items).item(),
        }
        if self.grad_norm_clipping is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm_clipping)
            logs["ir-grad-norm"] = grad_norm.item()
        self.optimizer.step()
        return logs

    def _agent_actions(self, batch: Batch, features: torch.Tensor) -> torch.Tensor:
        """
        Actions of shape `(*dims, n_agents)`, dropping the objective axis added by multi-objective batches.

        @ai-generated
        """
        actions = batch.actions
        while actions.ndim > features.ndim - 1:
            actions = actions[..., 0]
        return actions.long()

    def _time_step_masks(self, batch: Batch, features: torch.Tensor) -> torch.Tensor:
        """
        Validity of each time step, of shape `(*dims, 1)` so that it broadcasts over the agents.

        The masks are identical for every agent and objective, so the trailing axes added by
        `Batch.for_individual_learners` and by multi-objective batches are simply indexed away.

        @ai-generated
        """
        masks = batch.masks
        while masks.ndim > features.ndim - 2:
            masks = masks[..., 0]
        return masks.unsqueeze(-1).to(features.dtype)

    @classmethod
    def from_env(cls, env: MARLEnv[Any] | EnvConfig, n_features: int = 256, **kwargs):
        """Build an ICM whose encoder matches the observation space of `env`. @ai-generated"""
        match (env.observation_shape, env.extras_shape):
            case ((obs_size,), (n_extras,)):  # Linear
                encoder = model_bank.MLP((n_features,), obs_size, n_extras)
            case ((_, _, _) as dimensions, (n_extras,)):  # CNN
                encoder = model_bank.QCNN(n_features, env.n_agents, dimensions, (n_extras,), duelling=False)
            case other:
                raise ValueError(f"Unsupported (observation, extras) shape: {other}")
        return cls(encoder, env.n_agents, env.n_actions, n_features=n_features, **kwargs)

    def __hash__(self) -> int:
        # Required because the @dataclass decorator sets __hash__ to None when it generates __eq__.
        return id(self)
