import os
import torch

from .interfaces import LinearNN, RecurrentNN


class MLP(LinearNN):
    """
    Multi layer perceptron
    """

    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> None:
        assert len(input_shape) == 1, "MLP can only handle 1D inputs"
        assert len(extras_shape) == 1, "MLP can only handle 1D extras"
        assert len(output_shape) == 1, "MLP can only handle 1D outputs"
        super().__init__(input_shape, extras_shape, output_shape)
        input_size = input_shape[0] + extras_shape[0]
        output_size = output_shape[0]
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_size)
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor|None = None) -> torch.Tensor:
        if extras is not None:
            obs = torch.concat((obs, extras), dim=-1)
        return self.nn(obs)


class RNNQMix(RecurrentNN):
    """RNN used in the QMix paper"""

    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> None:
        assert len(input_shape) == 1, "RNNQMix can only handle 1D inputs"
        assert len(extras_shape) == 1, "RNNQMix can only handle 1D extras"
        assert len(output_shape) == 1, "RNNQMix can only handle 1D outputs"
        super().__init__(input_shape, extras_shape, output_shape)
        n_inputs = input_shape[0] + extras_shape[0]
        n_outputs = output_shape[0]
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, 64),
            torch.nn.ReLU()
        )
        self.gru = torch.nn.GRU(input_size=64, hidden_size=64)
        self.fc2 = torch.nn.Linear(64, n_outputs)

    def forward(
        self,
        obs: torch.Tensor,
        extras: torch.Tensor|None = None,
        hidden_states: torch.Tensor|None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(obs.shape) >= 3, "The observation should have shape (ep_length, batch_size, n_agents, obs_size)"
        assert obs.shape[:2] == extras.shape[:2]
        assert hidden_states is None or hidden_states.shape[:2] == obs.shape[:2]
        obs = torch.concat((obs, extras), dim=-1)
        x = self.fc1(obs)
        x, hidden_state = self.gru.forward(x, hidden_states)
        x = self.fc2(x)
        return x, hidden_state


class XtractMLP(LinearNN):
    """Self supervised (frozen) CNN feature extractor followed by an MLP"""
    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> None:
        super().__init__(input_shape, extras_shape, output_shape)
        self.embedding_shape = (1024, )
        file_directory = os.path.dirname(__file__)
        saved_model = os.path.join(file_directory, "saved_models/cnn_ssl_feature_extractor.model")
        self.cnn = CNN(input_shape, None, self.embedding_shape)
        with open(saved_model, "rb") as f:
            state_dict = torch.load(f)
            self.cnn.load_state_dict(state_dict)
        self.mlp = MLP(self.embedding_shape, extras_shape, output_shape)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor|None = None) -> torch.Tensor:
        obs = obs / 255.
        batch_size, n_agents, channels, height, width = obs.shape
        obs = obs.view(batch_size * n_agents, channels, height, width)
        features = self.cnn.forward(obs).detach()
        features = features.view(batch_size, n_agents, *self.embedding_shape)
        return self.mlp.forward(features, extras)


class AtariCNN(LinearNN):
    """The CNN used in the 2015 Mhin et al. DQN paper"""

    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...]|None, output_shape: tuple[int, ...]) -> None:
        assert len(input_shape) == 3
        assert len(output_shape) == 1
        super().__init__(input_shape, extras_shape, output_shape)
        filters = [32, 64, 64]
        kernels = [8, 4, 3]
        strides = [4, 2, 1]
        self.cnn, n_features = make_cnn(input_shape, filters, kernels, strides)
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_features, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, output_shape[0])
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor|None = None) -> torch.Tensor:
        batch_size, n_agents, channels, height, width = obs.shape
        obs = obs.view(batch_size * n_agents, channels, height, width)
        qvalues: torch.Tensor = self.nn.forward(obs)
        return qvalues.view(batch_size, n_agents, -1)



class CNN(LinearNN):
    """
    CNN with three convolutional layers. The CNN output (output_cnn) is flattened and the extras are
    concatenated to this output. The CNN is followed by three linear layers (512, 256, output_shape[0]).
    """

    def __init__(self, input_shape: tuple[int, int, int], extras_shape: tuple[int]|None, output_shape: tuple[int]) -> None:
        assert len(input_shape) == 3, f"CNN can only handle 3D input shapes ({len(input_shape)} here)"
        assert extras_shape is None or len(extras_shape) == 1, f"CNN can only handle 1D extras shapes ({len(extras_shape)} here)"
        assert len(output_shape) == 1, f"CNN can only handle 1D input shapes ({len(output_shape)} here)"
        super().__init__(input_shape, extras_shape, output_shape)
        
        num_extras = extras_shape[0] if extras_shape is not None else 0
        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        filters = [32, 64, 64]

        self.cnn, n_features = make_cnn(input_shape, filters, kernel_sizes, strides)
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_features + num_extras, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_shape[0])
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor = None) -> torch.Tensor:
        # Check that the input has the correct shape (at most 4 dimensions)
        *dims, channels, height, width = obs.shape
        obs = obs.view(-1, channels, height, width)
        extras = extras.view(-1, *self.extras_shape)
        features = self.cnn.forward(obs)
        if extras is not None:
            features = torch.concat((features, extras), dim=-1)
        res: torch.Tensor = self.linear(features)
        return res.view(*dims, *self.output_shape)


class PolicyNetworkMLP(LinearNN):
    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...] | None, output_shape: tuple[int, ...]):
        assert len(extras_shape) == 1 and len(output_shape) == 1 and len(input_shape) == 1
        super().__init__(input_shape, extras_shape, output_shape)
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_shape[0] + extras_shape[0], 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_shape[0]),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor|None = None) -> torch.Tensor:
        if extras is not None:
            obs = torch.cat((obs, extras), dim=-1)
        return self.nn.forward(obs)


def make_cnn(input_shape, filters: list[int], kernel_sizes: list[int], strides: list[int]):
    """Create a CNN with flattened output based on the given filters, kernel sizes and strides."""
    channels, height, width = input_shape
    paddings = [0 for _ in filters]
    n_padded = 0
    output_w, output_h = conv2d_size_out(width, height, kernel_sizes, strides, paddings)
    while output_w < 0 or output_h < 0:
        # Add paddings if the output size is negative
        paddings[n_padded % len(paddings)] += 1
        n_padded += 1
        output_w, output_h = conv2d_size_out(width, height, kernel_sizes, strides, paddings)
    print("Padding added: ", paddings)
    assert output_h > 0 and output_w > 0, f"Input size = {input_shape}, output witdh = {output_w}, output height = {output_h}"
    modules = []
    for f, k, s, p in zip(filters, kernel_sizes, strides, paddings):
        modules.append(torch.nn.Conv2d(in_channels=channels, out_channels=f, kernel_size=k, stride=s, padding=p))
        modules.append(torch.nn.ReLU())
        channels = f
    modules.append(torch.nn.Flatten())
    output_size = output_h * output_w * filters[-1]
    return torch.nn.Sequential(*modules), output_size


def conv2d_size_out(input_width, input_height, kernel_sizes, strides, paddings):
    """
    Compute the output width and height of a sequence of 2D convolutions.
    See shape section on https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    width = input_width
    height = input_height
    for kernel_size, stride, pad  in zip(kernel_sizes, strides, paddings):
        width = (width + 2*pad - (kernel_size - 1) - 1) // stride + 1
        height = (height + 2*pad - (kernel_size - 1) - 1) // stride + 1
    return width, height
