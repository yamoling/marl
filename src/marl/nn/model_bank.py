from typing import Optional
from dataclasses import dataclass
from rlenv.models import RLEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.distributions import kl_divergence
from torch.autograd import Variable
import math
from functools import reduce
import operator
from marl.models.nn import QNetwork, RecurrentQNetwork, ActorCriticNN, NN, MAICNN


@dataclass(unsafe_hash=True)
class MLP(QNetwork):
    """
    Multi layer perceptron
    """

    layer_sizes: tuple[int, ...]

    def __init__(
        self,
        input_size: int,
        extras_size: int,
        hidden_sizes: tuple[int, ...],
        output_size: int,
    ):
        super().__init__((input_size,), (extras_size,), (output_size,))
        self.layer_sizes = (input_size + extras_size, *hidden_sizes, output_size)
        layers = [torch.nn.Linear(input_size + extras_size, hidden_sizes[0]), torch.nn.ReLU()]
        for i in range(len(hidden_sizes) - 1):
            layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_sizes[-1], output_size))
        self.nn = torch.nn.Sequential(*layers)

    @classmethod
    def from_env(cls, env: RLEnv, hidden_sizes: Optional[tuple[int, ...]] = None):
        if hidden_sizes is None:
            hidden_sizes = (64,)
        return MLP(
            env.observation_shape[0],
            env.extra_feature_shape[0],
            hidden_sizes,
            env.n_actions,
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        obs = torch.concat((obs, extras), dim=-1)
        return self.nn(obs)


class RNNQMix(RecurrentQNetwork):
    """RNN used in the QMix paper:
    - linear 64
    - relu
    - GRU 64
    - relu
    - linear 64"""

    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> None:
        assert len(input_shape) == 1, "RNNQMix can only handle 1D inputs"
        assert len(extras_shape) == 1, "RNNQMix can only handle 1D extras"
        assert len(output_shape) == 1, "RNNQMix can only handle 1D outputs"
        super().__init__(input_shape, extras_shape, output_shape)
        n_inputs = input_shape[0] + extras_shape[0]
        n_outputs = output_shape[0]
        self.fc1 = torch.nn.Sequential(torch.nn.Linear(n_inputs, 64), torch.nn.ReLU())
        self.gru = torch.nn.GRU(input_size=64, hidden_size=64, batch_first=False)
        self.fc2 = torch.nn.Linear(64, n_outputs)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor | None = None):
        self.gru.flatten_parameters()
        assert len(obs.shape) >= 3, "The observation should have at least shape (ep_length, batch_size, obs_size)"
        # During batch training, the input has shape (episodes_length, batch_size, n_agents, obs_size).
        # This shape is not supported by the GRU layer, so we merge the batch_size and n_agents dimensions
        # while keeping the episode_length dimension.
        episode_length, *batch_agents, obs_size = obs.shape
        obs = torch.reshape(obs, (episode_length, -1, obs_size))
        if extras is not None:
            extras = torch.reshape(extras, (*obs.shape[:-1], *self.extras_shape))
            obs = torch.concat((obs, extras), dim=-1)
        x = self.fc1.forward(obs)
        x, self.hidden_states = self.gru.forward(x, self.hidden_states)
        x = self.fc2.forward(x)
        # Restore the original shape of the batch
        x = x.view(episode_length, *batch_agents, *self.output_shape)
        return x


RNN = RNNQMix


class DuelingMLP(QNetwork):
    def __init__(self, nn: QNetwork, output_size: int):
        assert len(nn.output_shape) == 1
        super().__init__(nn.input_shape, nn.extras_shape, (output_size,))
        self.nn = nn
        self.value_head = torch.nn.Linear(nn.output_shape[0], 1)
        self.advantage = torch.nn.Linear(nn.output_shape[0], output_size)

    @classmethod
    def from_env(cls, env: RLEnv, nn: QNetwork):
        assert nn.input_shape == env.observation_shape
        assert nn.extras_shape == env.extra_feature_shape
        return cls(nn, env.n_actions)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        features = self.nn.forward(obs, extras)
        features = torch.nn.functional.relu(features)
        value = self.value_head.forward(features)
        advantage = self.advantage.forward(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class AtariCNN(QNetwork):
    """The CNN used in the 2015 Mhin et al. DQN paper"""

    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], output_shape: tuple[int, ...]):
        assert len(input_shape) == 3
        assert len(output_shape) == 1
        super().__init__(input_shape, extras_shape, output_shape)
        filters = [32, 64, 64]
        kernels = [8, 4, 3]
        strides = [4, 2, 1]
        self.cnn, n_features = make_cnn(input_shape, filters, kernels, strides)
        self.linear = torch.nn.Sequential(torch.nn.Linear(n_features, 512), torch.nn.ReLU(), torch.nn.Linear(512, output_shape[0]))

    def forward(self, obs: torch.Tensor, extras: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, n_agents, channels, height, width = obs.shape
        obs = obs.view(batch_size * n_agents, channels, height, width)
        qvalues: torch.Tensor = self.cnn.forward(obs)
        return qvalues.view(batch_size, n_agents, -1)


class CNN(QNetwork):
    """
    CNN with three convolutional layers. The CNN output (output_cnn) is flattened and the extras are
    concatenated to this output. The CNN is followed by three linear layers (512, 256, output_shape[0]).
    """

    def __init__(self, input_shape: tuple[int, ...], extras_size: int, output_size: int):
        assert len(input_shape) == 3, f"CNN can only handle 3D input shapes ({len(input_shape)} here)"
        super().__init__(input_shape, (extras_size,), (output_size,))

        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        filters = [32, 64, 64]

        self.cnn, n_features = make_cnn(self.input_shape, filters, kernel_sizes, strides)
        self.linear = MLP(n_features, extras_size, (64, 64), output_size)

    @classmethod
    def from_env(cls, env: RLEnv):
        return cls(env.observation_shape, env.extra_feature_shape[0], env.n_actions)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        # For transitions, the shape is (batch_size, n_agents, channels, height, width)
        # For episodes, the shape is (time, batch_size, n_agents, channels, height, width)
        *dims, channels, height, width = obs.shape
        leading_dims_size = math.prod(dims)
        # We must use 'reshape' instead of 'view' to handle the case of episodes
        obs = obs.view(leading_dims_size, channels, height, width).to(self.device)
        features = self.cnn.forward(obs)
        extras = extras.view(leading_dims_size, *self.extras_shape).to(self.device)
        res = self.linear.forward(features, extras)
        return res.view(*dims, *self.output_shape)

class CommCNN(CNN):
    @classmethod
    def from_env(cls, env: RLEnv, channel_size: int):
        return cls(env.observation_shape, env.extra_feature_shape[0] + channel_size, env.n_actions)

class CNN_ActorCritic(ActorCriticNN):
    def __init__(self, input_shape: tuple[int, int, int], extras_shape: tuple[int], output_shape: tuple[int]):
        assert len(input_shape) == 3, f"CNN can only handle 3D input shapes ({len(input_shape)} here)"
        assert extras_shape is None or len(extras_shape) == 1, f"CNN can only handle 1D extras shapes ({len(extras_shape)} here)"
        assert len(output_shape) == 1, f"CNN can only handle 1D input shapes ({len(output_shape)} here)"
        super().__init__(input_shape, extras_shape, output_shape)

        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        filters = [32, 64, 64]

        self.cnn, n_features = make_cnn(self.input_shape, filters, kernel_sizes, strides)
        common_input_size = n_features + self.extras_shape[0]
        self.common = torch.nn.Sequential(
            torch.nn.Linear(common_input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
        )

        self.policy_network = torch.nn.Sequential(
            torch.nn.Linear(128, *output_shape),
            # torch.nn.Softmax(dim=-1), use logits to mask invalid actions
        )
        self.value_network = torch.nn.Linear(128, 1)

    def _cnn_forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Check that the input has the correct shape (at least 4 dimensions)
        *dims, channels, height, width = obs.shape
        obs = obs.view(-1, channels, height, width)
        features = self.cnn.forward(obs)
        return features

    def policy(self, obs: torch.Tensor):
        return self.policy_network(obs)

    def value(self, obs: torch.Tensor):
        return self.value_network(obs)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor):
        features = self._cnn_forward(obs)
        extras = extras.view(-1, *self.extras_shape)
        features = torch.cat((features, extras), dim=-1)
        x = self.common(features)
        return self.policy(x), self.value(x)

    @property
    def value_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.cnn.parameters()) + list(self.common.parameters()) + list(self.value_network.parameters())

    @property
    def policy_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.cnn.parameters()) + list(self.common.parameters()) + list(self.policy_network.parameters())


class SimpleActorCritic(ActorCriticNN):
    def __init__(self, input_shape: tuple[int], extras_shape: tuple[int, ...], output_shape: tuple[int]):
        super().__init__(input_shape, extras_shape, output_shape)
        self.common = torch.nn.Sequential(
            torch.nn.Linear(*input_shape, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.policy_network = torch.nn.Sequential(
            torch.nn.Linear(256, *output_shape),
            torch.nn.Softmax(dim=-1),
        )

        self.value_network = torch.nn.Linear(256, 1)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor):
        obs = torch.cat((obs, extras), dim=-1)
        x = self.common(obs)
        return self.policy_network(x), self.value_network(x)

    def policy(self, obs: torch.Tensor):
        return self.policy_network(obs)

    def value(self, obs: torch.Tensor):
        return self.value_network(obs)

    @property
    def value_parameters(self):
        return list(self.common.parameters()) + list(self.value_network.parameters())

    @property
    def policy_parameters(self):
        return list(self.common.parameters()) + list(self.policy_network.parameters())


class PolicyNetworkMLP(NN):
    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], output_shape: tuple[int, ...]):
        super().__init__(input_shape, extras_shape, output_shape)
        assert len(self.extras_shape) == 1 and len(output_shape) == 1 and len(input_shape) == 1
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_shape[0] + self.extras_shape[0], 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_shape[0]),
            torch.nn.Softmax(dim=-1),
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor | None = None) -> torch.Tensor:
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
    assert output_h > 0 and output_w > 0, f"Input size = {input_shape}, output witdh = {output_w}, output height = {output_h}"
    modules = []
    for f, k, s, p in zip(filters, kernel_sizes, strides, paddings):
        modules.append(torch.nn.Conv2d(in_channels=channels, out_channels=f, kernel_size=k, stride=s, padding=p))
        modules.append(torch.nn.ReLU())
        channels = f
    modules.append(torch.nn.Flatten())
    output_size = output_h * output_w * filters[-1]
    return torch.nn.Sequential(*modules), output_size


def conv2d_size_out(input_width: int, input_height: int, kernel_sizes: list[int], strides: list[int], paddings: list[int]):
    """
    Compute the output width and height of a sequence of 2D convolutions.
    See shape section on https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    width = input_width
    height = input_height
    for kernel_size, stride, pad in zip(kernel_sizes, strides, paddings):
        width = (width + 2 * pad - (kernel_size - 1) - 1) // stride + 1
        height = (height + 2 * pad - (kernel_size - 1) - 1) // stride + 1
    return width, height


class ACNetwork(ActorCriticNN):
    def __init__(self, input_shape: tuple[int], extras_shape: tuple, output_shape: tuple):
        super().__init__(input_shape, extras_shape, output_shape)
        print(input_shape)
        self.common = torch.nn.Sequential(
            torch.nn.Linear(*input_shape, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
        )
        self.policy_network = torch.nn.Linear(128, *output_shape)
        self.value_network = torch.nn.Linear(128, 1)

    def policy(self, obs: torch.Tensor):
        obs = self.common(obs)
        return self.policy_network(obs)

    def value(self, obs: torch.Tensor):
        obs = self.common(obs)
        return self.value_network(obs)

    def forward(self, x):
        x = self.common(x)
        return self.policy_network(x), self.value_network(x)

    @property
    def value_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.common.parameters()) + list(self.value_network.parameters())

    @property
    def policy_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.common.parameters()) + list(self.policy_network.parameters())


class ACNetwork2(ActorCriticNN):
    def __init__(self, input_shape: tuple[int], extras_shape: tuple, output_shape: tuple):
        super().__init__(input_shape, extras_shape, output_shape)
        self.value_network = torch.nn.Sequential(
            torch.nn.Linear(*input_shape, 128), torch.nn.ReLU(), torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1)
        )
        self.policy_network = torch.nn.Sequential(
            torch.nn.Linear(*input_shape, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, *output_shape),
        )

    def policy(self, obs: torch.Tensor):
        return self.policy_network(obs)

    def value(self, obs: torch.Tensor):
        return self.value_network(obs)

    def forward(self, x):
        return self.policy_network(x), self.value_network(x)

    @property
    def value_parameters(self):
        return self.value_network.parameters()

    @property
    def policy_parameters(self):
        return self.policy_network.parameters()


class CNet(NN): # Source : https://github.com/minqi/learning-to-communicate-pytorch
    def __init__(self, input_shape: tuple[int], extras_size: int, output_size: int, opt):
        assert len(input_shape) == 3, f"CNN can only handle 3D input shapes ({len(input_shape)} here)"
        super().__init__(input_shape, (extras_size,), (output_size,))

        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        filters = [32, 64, 64]

        self.cnn, n_features = make_cnn(self.input_shape, filters, kernel_sizes, strides)

        self.opt = opt
        self.comm_size = opt.game_comm_bits
        self.init_param_range = (-0.08, 0.08)

        # Set up inputs
        self.agent_lookup = nn.Embedding(opt.game_nagents, opt.model_rnn_size)
        self.state_lookup = nn.Linear(reduce(operator.mul, input_shape), opt.model_rnn_size) # TODO : verify that num_embeddings is n_features = input size
        # Action aware
        self.prev_message_lookup = None
        if opt.model_action_aware:
            if opt.model_dial:
                self.prev_action_lookup = nn.Embedding(opt.game_action_space_total, opt.model_rnn_size)
            else:
                self.prev_action_lookup = nn.Embedding(opt.game_action_space + 1, opt.model_rnn_size)
                self.prev_message_lookup = nn.Embedding(opt.game_comm_bits + 1, opt.model_rnn_size)
            
        # Communication enabled
        if opt.comm_enabled:
            self.messages_mlp = nn.Sequential()
            if opt.model_bn:
                self.messages_mlp.add_module('batchnorm1', nn.BatchNorm1d(self.comm_size))
            self.messages_mlp.add_module('linear1', nn.Linear(self.comm_size, opt.model_rnn_size))
            if opt.model_comm_narrow:
                self.messages_mlp.add_module('relu1', nn.ReLU(inplace=True))
                    
        # Set up RNN
        dropout_rate = opt.model_rnn_dropout_rate or 0 # TODO : set opt.model_rnn_dropout_rate
        self.rnn = nn.GRU(input_size=opt.model_rnn_size, hidden_size=opt.model_rnn_size, 
            num_layers=opt.model_rnn_layers, dropout=dropout_rate, batch_first=True)

        # Set up outputs
        self.outputs = nn.Sequential()
        if dropout_rate > 0:
            self.outputs.add_module('dropout1', nn.Dropout(dropout_rate))
        self.outputs.add_module('linear1', nn.Linear(opt.model_rnn_size, opt.model_rnn_size))
        if opt.model_bn:
            self.outputs.add_module('batchnorm1', nn.BatchNorm1d(opt.model_rnn_size))
        self.outputs.add_module('relu1', nn.ReLU(inplace=True))
        self.outputs.add_module('linear2', nn.Linear(opt.model_rnn_size, opt.game_action_space_total))

    def get_params(self):
        return list(self.parameters())

    def reset_parameters(self):
        opt = self.opt
        self.messages_mlp.linear1.reset_parameters()
        self.rnn.reset_parameters()
        self.agent_lookup.reset_parameters()
        self.state_lookup.reset_parameters()
        self.prev_action_lookup.reset_parameters()
        if self.prev_message_lookup:
            self.prev_message_lookup.reset_parameters()
        if opt.comm_enabled and opt.model_dial:
            self.messages_mlp.batchnorm1.reset_parameters()
        self.outputs.linear1.reset_parameters()
        self.outputs.linear2.reset_parameters()
        for p in self.rnn.parameters():
            p.data.uniform_(*self.init_param_range)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, messages, hidden, prev_action, agent_index):
        opt = self.opt

        # For transitions, the shape is (batch_size, n_agents, channels, height, width)
        # For episodes, the shape is (time, batch_size, n_agents, channels, height, width)
        # *dims, channels, height, width = obs.shape
        # leading_dims_size = math.prod(dims)
        # # We must use 'reshape' instead of 'view' to handle the case of episodes
        # obs = obs.reshape(leading_dims_size, channels, height, width).to(self.device)
        # features = self.cnn.forward(obs)
        # extras = extras.reshape(leading_dims_size, *self.extras_shape).to(self.device)
        # features = torch.concat((features, extras), dim=-1)
        # features = features[agent_index, :].squeeze(0)
        s_t = Variable(obs.squeeze(0)[agent_index])
        hidden = Variable(hidden)
        prev_message = None
        if opt.model_dial:
            if opt.model_action_aware:
                prev_action = Variable(prev_action)
        else:
            if opt.model_action_aware:
                prev_action, prev_message = prev_action
                prev_action = Variable(prev_action)
                prev_message = Variable(prev_message)
            messages = Variable(messages)
        agent_index = Variable(agent_index)

        z_a, z_o, z_u, z_m = [0]*4
        z_a = self.agent_lookup(agent_index)
        z_o = self.state_lookup(s_t.view(-1)) # 1D state
        if opt.model_action_aware:
            z_u = self.prev_action_lookup(prev_action)
            if prev_message is not None:
                z_u += self.prev_message_lookup(prev_message)
        z_m = self.messages_mlp(messages.view(-1, self.comm_size))
        z = z_a + z_o + z_u + z_m
        z = z.unsqueeze(1)

        hidden = hidden.squeeze(2)
        rnn_out, h_out = self.rnn(z, hidden)
        outputs = self.outputs(rnn_out[:, -1, :].squeeze())

        return h_out, outputs

    @classmethod
    def from_env(cls, env: RLEnv, opt):
        return cls(env.observation_shape, env.extra_feature_shape[0], opt.game_action_space_total, opt)


class MAICNetwork(MAICNN):
    def __init__(self, input_shape: tuple[int, ...], extras_size: int, output_size: int, args):
        assert len(input_shape) == 3, f"CNN can only handle 3D input shapes ({len(input_shape)} here)"
        super().__init__(input_shape, (extras_size,), (output_size,))
        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        filters = [32, 64, 64]
    
        self.cnn, n_features = make_cnn(self.input_shape, filters, kernel_sizes, strides)
        self.args = args
        self.n_agents = args.n_agents
        self.latent_dim = args.latent_dim
        self.n_actions = output_size

        NN_HIDDEN_SIZE = args.nn_hidden_size
        activation_func = nn.LeakyReLU()

        self.embed_net = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, NN_HIDDEN_SIZE),
            nn.BatchNorm1d(NN_HIDDEN_SIZE),
            activation_func,
            nn.Linear(NN_HIDDEN_SIZE, args.n_agents * args.latent_dim * 2)
        )

        self.inference_net = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim + self.n_actions, NN_HIDDEN_SIZE),
            nn.BatchNorm1d(NN_HIDDEN_SIZE),
            activation_func,
            nn.Linear(NN_HIDDEN_SIZE, args.latent_dim * 2)
        )

        self.fc1 = nn.Linear(n_features + extras_size, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, self.n_actions)
        
        self.msg_net = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim + args.latent_dim, NN_HIDDEN_SIZE),
            activation_func,
            nn.Linear(NN_HIDDEN_SIZE, self.n_actions)
        )

        self.w_query = nn.Linear(args.rnn_hidden_dim, args.attention_dim)
        self.w_key = nn.Linear(args.latent_dim, args.attention_dim)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, hidden_state, test_mode=False):
        # For transitions, the shape is (batch_size, n_agents, channels, height, width)
        # For episodes, the shape is (time, batch_size, n_agents, channels, height, width)
        *dims, channels, height, width = obs.shape
        bs = dims[0]
        leading_dims_size = math.prod(dims)
        # We must use 'reshape' instead of 'view' to handle the case of episodes
        obs = obs.reshape(leading_dims_size, channels, height, width).to(self.device)
        features = self.cnn.forward(obs)
        extras = extras.reshape(leading_dims_size, *self.extras_shape).to(self.device)
        features = torch.concat((features, extras), dim=-1)

        x = F.relu(self.fc1(features))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        latent_parameters = self.embed_net(h)
        latent_parameters[:, -self.n_agents * self.latent_dim:] = torch.clamp(
            torch.exp(latent_parameters[:, -self.n_agents * self.latent_dim:]),
            min=self.args.var_floor)

        latent_embed = latent_parameters.reshape(bs * self.n_agents, self.n_agents * self.latent_dim * 2)

        if test_mode:
            latent = latent_embed[:, :self.n_agents * self.latent_dim]
        else:
            gaussian_embed = D.Normal(latent_embed[:, :self.n_agents * self.latent_dim],
                                    (latent_embed[:, self.n_agents * self.latent_dim:]) ** (1 / 2))
            latent = gaussian_embed.rsample() # shape: (bs * self.n_agents, self.n_agents * self.latent_dim)
        latent = latent.reshape(bs * self.n_agents * self.n_agents, self.latent_dim)

        h_repeat = h.view(bs, self.n_agents, -1).repeat(1, self.n_agents, 1).view(bs * self.n_agents * self.n_agents, -1)
        msg = self.msg_net(torch.cat([h_repeat, latent], dim=-1)).view(bs, self.n_agents, self.n_agents, self.n_actions)
        
        query = self.w_query(h).unsqueeze(1)
        key = self.w_key(latent).reshape(bs * self.n_agents, self.n_agents, -1).transpose(1, 2)
        alpha = torch.bmm(query / (self.args.attention_dim ** (1/2)), key).view(bs, self.n_agents, self.n_agents)
        for i in range(self.n_agents):
            alpha[:, i, i] = -1e9
        alpha = F.softmax(alpha, dim=-1).reshape(bs, self.n_agents, self.n_agents, 1)

        if test_mode:
            alpha[alpha < (0.25 * 1 / self.n_agents)] = 0

        gated_msg = alpha * msg

        return_q = q + torch.sum(gated_msg, dim=1).view(bs * self.n_agents, self.n_actions)

        returns = {}
    
        #if self.args.mi_loss_weight > 0:
        returns['mi_loss'] = self.calculate_action_mi_loss(h, bs, latent_embed, return_q)
        #if self.args.entropy_loss_weight > 0:
        query = self.w_query(h.detach()).unsqueeze(1)
        key = self.w_key(latent.detach()).reshape(bs * self.n_agents, self.n_agents, -1).transpose(1, 2)
        alpha = F.softmax(torch.bmm(query, key), dim=-1).reshape(bs, self.n_agents, self.n_agents)
        returns['entropy_loss'] = self.calculate_entropy_loss(alpha)

        return return_q, h, returns

    def calculate_action_mi_loss(self, h, bs, latent_embed, q):
        latent_embed = latent_embed.view(bs * self.n_agents, 2, self.n_agents, self.latent_dim)
        g1 = D.Normal(latent_embed[:, 0, :, :].reshape(-1, self.latent_dim), latent_embed[:, 1, :, :].reshape(-1, self.latent_dim) ** (1/2))
        hi = h.view(bs, self.n_agents, 1, -1).repeat(1, 1, self.n_agents, 1).view(bs * self.n_agents * self.n_agents, -1)
        
        selected_action = torch.max(q, dim=1)[1].unsqueeze(-1)
        one_hot_a = torch.zeros(selected_action.shape[0], self.n_actions).to(self.device).scatter(1, selected_action, 1)
        one_hot_a = one_hot_a.view(bs, 1, self.n_agents, -1).repeat(1, self.n_agents, 1, 1)
        one_hot_a = one_hot_a.view(bs * self.n_agents * self.n_agents, -1)

        latent_infer = self.inference_net(torch.cat([hi, one_hot_a], dim=-1)).view(bs * self.n_agents * self.n_agents, -1)
        latent_infer[:, self.latent_dim:] = torch.clamp(torch.exp(latent_infer[:, self.latent_dim:]), min=self.args.var_floor)
        g2 = D.Normal(latent_infer[:, :self.latent_dim], latent_infer[:, self.latent_dim:] ** (1/2))
        mi_loss = kl_divergence(g1, g2).sum(-1).mean()
        return mi_loss * self.args.mi_loss_weight

    def calculate_entropy_loss(self, alpha):
        alpha = torch.clamp(alpha, min=1e-4)
        entropy_loss = - (alpha * torch.log2(alpha)).sum(-1).mean()
        return entropy_loss * self.args.entropy_loss_weight

    @classmethod
    def from_env(cls, env: RLEnv, args):
        return cls(env.observation_shape, env.extra_feature_shape[0], env.n_actions, args)