from dataclasses import dataclass
import torch
from marlenv.utils import Schedule
from marlenv import MARLEnv
from marl.models import IRModule, Batch, NN
from marl.nn import model_bank


@dataclass
class ICM(IRModule, NN):
    """
    Intrinsic Curiosity Module (ICM) for multi-agent reinforcement learning and discrete action spaces.

    Paper: https://arxiv.org/pdf/1705.05363
    """

    weight: Schedule
    n_agents: int
    n_actions: int
    n_features: int

    def __init__(self, feature_encoder: NN, n_agents: int, n_actions: int, weight: float | Schedule = 0.01, n_features: int = 256):
        IRModule.__init__(self)
        NN.__init__(self, feature_encoder.input_shape, feature_encoder.extras_shape, (n_features,))

        features_shape = feature_encoder.output_shape
        assert len(features_shape) == 1, "Feature encoder must output a single feature vector"
        self.n_features = features_shape[0]
        self.n_agents = n_agents
        self.n_actions = n_actions
        if isinstance(weight, (float, int)):
            weight = Schedule.constant(weight)
        self.weight = weight

        # Feature encoder s → φ(s)
        self._feature = feature_encoder

        # Inverse model: φ(s), φ(s') → a (one-hot encoded action for each agent)
        self._inverse_model = torch.nn.Sequential(
            torch.nn.Linear(self.n_features * 2, n_features),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(n_features, n_features),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(n_features, n_agents * n_actions),
        )

        # Forward model: φ(s), a → φ(s')
        self._forward_model = torch.nn.Sequential(
            torch.nn.Linear(self.n_actions * self.n_agents + self.n_features, n_features),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(n_features, n_features),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(n_features, self.n_features),
        )

        self._optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self._cross_entropy = torch.nn.CrossEntropyLoss()
        self._mse_loss = torch.nn.MSELoss()

    def forward(self, batch: Batch) -> torch.Tensor:
        with torch.no_grad():
            features = self._feature.forward(batch.states, batch.states_extras)
            next_features = self._feature.forward(batch.next_states, batch.next_states_extras)
            one_hot_actions = batch.one_hot_actions.view(batch.size, -1)
            forward_inputs = torch.cat((features, one_hot_actions), 1)
            next_features_pred = self._forward_model(forward_inputs)
            # Equation (6) in the paper: $r_i = \frac{\eta}{2} \times ||\hat{φ}(s') - φ(s')||^2_2$
            intrinsic_reward = self.weight / 2 * torch.norm(next_features_pred - next_features, dim=1).pow(2)
        return intrinsic_reward

    def compute(self, batch: Batch):
        return self.forward(batch)

    def update(self, batch: Batch, time_step: int) -> dict[str, float]:
        self.weight.update(time_step)

        # Feature computation
        features = self._feature.forward(batch.states, batch.states_extras)
        next_features = self._feature.forward(batch.next_states, batch.next_states_extras)

        # Inverse model loss
        inverse_inputs = torch.cat((features, next_features), 1)
        predicted_actions = self._inverse_model.forward(inverse_inputs)
        predicted_actions = torch.reshape(predicted_actions, (batch.size, self.n_agents, self.n_actions))
        predicted_action_probs = torch.nn.functional.softmax(predicted_actions, dim=-1)
        predicted_action_probs = predicted_action_probs.view(batch.size * self.n_agents, self.n_actions)
        ground_truth = batch.actions.flatten()
        inverse_loss = self._cross_entropy.forward(predicted_action_probs, ground_truth)

        # Forward model loss
        one_hot_actions = batch.one_hot_actions.view(batch.size, -1)
        forward_inputs = torch.cat((features, one_hot_actions), 1)
        next_features_pred = self._forward_model(forward_inputs)
        forward_loss = self._mse_loss.forward(next_features_pred, next_features)

        # Total loss
        loss = inverse_loss + forward_loss
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

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
                nn = model_bank.MLP(
                    size,
                    n_extras,
                    (128, 256, 128),
                    output_shape,
                )
            case ((_, _, _) as dimensions, (n_extras,)):  # CNN
                nn = model_bank.CNN(
                    dimensions,
                    n_extras,
                    output_shape,
                )
            case other:
                raise ValueError(f"Unsupported (obs, extras) shape: {other}")
        return ICM(nn, env.n_agents, env.n_actions, n_features=n_features)

    def __hash__(self) -> int:
        # Required for deserialization (in torch.nn.module)
        return hash(self.name)
