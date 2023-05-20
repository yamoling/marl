import torch
import os

from .mixer import Mixer


class AbsLayer(torch.nn.Module):
    def forward(self, x):
        return torch.abs(x)


class ReshapeLayer(torch.nn.Module):
    def __init__(self, *output_shape: tuple[int]) -> None:
        super().__init__()
        self._output_shape = output_shape

    def forward(self, x: torch.Tensor):
        return x.reshape(*self._output_shape)


class LambdaLayer(torch.nn.Module):
    def __init__(self, function) -> None:
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


class BMMLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weights: torch.FloatTensor = None
        self.biases: torch.FloatTensor = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bmm(x, self.weights) + self.biases


class HyperNetwork(torch.nn.Module):
    def __init__(self, state_size: int, n_agents: int, n_hidden_features: int) -> None:
        super().__init__()
        num_weights1 = n_agents * n_hidden_features
        num_weights2 = n_hidden_features * 1
        num_layers = 2
        if num_layers == 2:
            self.w1 = torch.nn.Sequential(
                torch.nn.Linear(state_size, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, num_weights1),
            )
            self.w2 = torch.nn.Sequential(
                torch.nn.Linear(state_size, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, num_weights2),
            )
        else:
            self.w1 = torch.nn.Sequential(torch.nn.Linear(state_size, num_weights1))
            self.w2 = torch.nn.Sequential(torch.nn.Linear(state_size, num_weights2))
        self.w1.append(AbsLayer())
        self.w1.append(ReshapeLayer(-1, n_agents, n_hidden_features))
        self.w2.append(AbsLayer())
        self.w2.append(ReshapeLayer(-1, n_hidden_features, 1))

        self.b1 = torch.nn.Sequential(
            torch.nn.Linear(state_size, n_hidden_features),
            ReshapeLayer(-1, 1, n_hidden_features)
        )
        self.b2 = torch.nn.Sequential(
            torch.nn.Linear(state_size, 1),
            ReshapeLayer(-1, 1, 1)
        )

    def forward(self, states: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        w1 = self.w1(states)
        b1 = self.b1(states)
        w2 = self.w2(states)
        b2 = self.b2(states)
        return [(w1, b1), (w2, b2)]


class QMix(Mixer):
    """QMix mixer"""

    def __init__(self, state_size: int, n_agents: int, hidden_dims: int) -> None:
        super().__init__(n_agents)
        self.hyper_network = HyperNetwork(state_size, n_agents, hidden_dims)
        self._state_size = state_size
        self._n_agents = n_agents
        self._hidden_dims = hidden_dims

        self.fc1 = BMMLayer()
        self.fc2 = BMMLayer()
        self.estimator = torch.nn.Sequential(
            self.fc1,
            torch.nn.ELU(),
            self.fc2,
        )

    def _set_weights(self, weights: list[tuple[torch.Tensor, torch.Tensor]]):
        i = 0
        for layer in self.estimator:
            if isinstance(layer, BMMLayer):
                layer: BMMLayer = layer
                w, b = weights[i]
                layer.weights = w
                layer.biases = b
                i += 1

    def save(self, to_directory: str):
        filename = os.path.join(to_directory, "qmix.weights")
        return torch.save(self.state_dict(), filename)
    
    def load(self, from_directory: str):
        filename = os.path.join(from_directory, "qmix.weights")
        return self.load_state_dict(torch.load(filename))

    def forward(self, qvalues: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        # episode_length, batch_size, n_agents = qvalues.shape
        *dims, n_agents = qvalues.shape
        states = torch.reshape(states, (-1, self._state_size))
        qvalues = torch.reshape(qvalues, (-1, 1, n_agents))
        weights = self.hyper_network.forward(states)
        self._set_weights(weights)
        mixed_qvalues = self.estimator.forward(qvalues)
        return torch.reshape(mixed_qvalues, (*dims, ))


    def summary(self) -> dict[str, ]:
        return {
            **super().summary(),
            "state_size": self._state_size,
            "n_agents": self._n_agents,
            "hidden_dims": self._hidden_dims
        }