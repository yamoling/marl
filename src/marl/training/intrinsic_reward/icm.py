from dataclasses import KW_ONLY, dataclass, field

import torch
from marlenv import MARLEnv
from marlenv.utils import Schedule

from marl.models import NN, Batch, IRModule
from marl.nn import model_bank


@dataclass
class ICM(IRModule):
    """
    Intrinsic Curiosity Module (ICM) for multi-agent reinforcement learning and discrete action spaces.

    Paper: https://arxiv.org/pdf/1705.05363
    """

    feature_encoder: NN
    """Feature encoder s → φ(s)"""
    n_agents: int
    n_actions: int
    _: KW_ONLY
    output_shape: tuple[int, ...] = (1,)
    n_features: int = 256
    weight: Schedule = field(default_factory=lambda: Schedule.constant(0.01))

    def __post_init__(self):
        super().__post_init__()
        if self.output_size > 1:
            raise ValueError("ICM does not support multi-objective RL")

        # Inverse model: φ(s), φ(s') → a (one-hot encoded action for each agent)
        self.inverse_model = torch.nn.Sequential(
            torch.nn.Linear(self.n_features * 2, self.n_features),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.n_features, self.n_features),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.n_features, self.n_agents * self.n_actions),
        )

        # Forward model: φ(s), a → φ(s')
        self.forward_model = torch.nn.Sequential(
            torch.nn.Linear(self.n_actions * self.n_agents + self.n_features, self.n_features),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.n_features, self.n_features),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.n_features, self.n_features),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.mse_loss = torch.nn.MSELoss()

    def to(self, device: torch.device, *args, **kwargs):
        self._feature.to(device)
        self.inverse_model.to(device)
        self.forward_model.to(device)
        return self

    def forward(self, batch: Batch) -> torch.Tensor:
        with torch.no_grad():
            features = self.feature_encoder.forward(batch.states, batch.states_extras)
            next_features = self.feature_encoder.forward(batch.next_states, batch.next_states_extras)
            one_hot_actions = batch.one_hot_actions.view(batch.size, -1)
            forward_inputs = torch.cat((features, one_hot_actions), 1)
            next_features_pred = self.forward_model(forward_inputs)
            # Equation (6) in the paper: $r_i = \frac{\eta}{2} \times ||\hat{φ}(s') - φ(s')||^2_2$
            intrinsic_reward = self.weight / 2 * torch.norm(next_features_pred - next_features, dim=1).pow(2)
        return intrinsic_reward

    def compute(self, batch: Batch):
        return self.forward(batch)

    def update(self, batch: Batch, time_step: int) -> dict[str, float]:
        self.weight.update(time_step)

        # Feature computation
        features = self.feature_encoder.forward(batch.states, batch.states_extras)
        next_features = self.feature_encoder.forward(batch.next_states, batch.next_states_extras)

        # Inverse model loss
        inverse_inputs = torch.cat((features, next_features), 1)
        predicted_actions = self.inverse_model.forward(inverse_inputs)
        predicted_actions = torch.reshape(predicted_actions, (batch.size, self.n_agents, self.n_actions))
        predicted_action_probs = torch.nn.functional.softmax(predicted_actions, dim=-1)
        predicted_action_probs = predicted_action_probs.view(batch.size * self.n_agents, self.n_actions)
        ground_truth = batch.actions.flatten()
        inverse_loss = self.cross_entropy.forward(predicted_action_probs, ground_truth)

        # Forward model loss
        one_hot_actions = batch.one_hot_actions.view(batch.size, -1)
        forward_inputs = torch.cat((features, one_hot_actions), 1)
        next_features_pred = self.forward_model(forward_inputs)
        forward_loss = self.mse_loss.forward(next_features_pred, next_features)

        # Total loss
        loss = inverse_loss + forward_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "ir-weight": self.weight.value,
            "icm-inverse-loss": inverse_loss.item(),
            "icm-forward-loss": forward_loss.item(),
            "ir-loss": loss.item(),
        }

    @staticmethod
    def from_env(env: MARLEnv, n_features: int = 256):
        if env.reward_space.size == 1:
            output_shape = (n_features,)
        else:
            output_shape = (*env.reward_space.shape, n_features)
        match (env.state_shape, env.state_extra_shape):
            case ((size,), (n_extras,)):  # Linear
                nn = model_bank.generic.MLP(output_shape, size, n_extras)
            case ((_, _, _) as dimensions, (n_extras,)):  # CNN
                nn = model_bank.CNN(output_shape, dimensions, n_extras)
            case other:
                raise ValueError(f"Unsupported (obs, extras) shape: {other}")
        return ICM(nn, env.n_agents, env.n_actions, n_features=n_features)

    def __hash__(self) -> int:
        # Required for deserialization (in torch.nn.module)
        return hash(self.name)
