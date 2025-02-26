import torch
import torch.nn as nn
from marlenv.models import MARLEnv

from marl.models import NN


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


class CNetNN(NN):  # Source : https://github.com/minqi/learning-to-communicate-pytorch ### Not working
    def __init__(self, input_shape: tuple[int], extras_shape: tuple[int], output_size: int, opt):
        super().__init__(input_shape, extras_shape, (output_size,))

        self.opt = opt
        self.comm_size = opt.game_comm_bits
        self.init_param_range = (-0.08, 0.08)

        # Set up inputs
        self.agent_lookup = nn.Embedding(opt.game_nagents, opt.model_rnn_size)
        self.state_lookup = nn.Linear(input_shape[0] + extras_shape[0], opt.model_rnn_size)
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
                self.messages_mlp.add_module("batchnorm1", nn.BatchNorm1d(self.comm_size))
            self.messages_mlp.add_module("linear1", nn.Linear(self.comm_size, opt.model_rnn_size))
            if opt.model_comm_narrow:
                self.messages_mlp.add_module("relu1", nn.ReLU(inplace=True))

        # Set up RNN
        dropout_rate = opt.model_rnn_dropout_rate or 0  # TODO : set opt.model_rnn_dropout_rate
        self.rnn = nn.GRU(
            input_size=opt.model_rnn_size,
            hidden_size=opt.model_rnn_size,
            num_layers=opt.model_rnn_layers,
            dropout=dropout_rate,
            batch_first=True,
        )

        # Set up outputs
        self.outputs = nn.Sequential()
        if dropout_rate > 0:
            self.outputs.add_module("dropout1", nn.Dropout(dropout_rate))
        self.outputs.add_module("linear1", nn.Linear(opt.model_rnn_size, opt.model_rnn_size))
        if opt.model_bn:
            self.outputs.add_module("batchnorm1", nn.BatchNorm1d(opt.model_rnn_size))
        self.outputs.add_module("relu1", nn.ReLU(inplace=True))
        self.outputs.add_module("linear2", nn.Linear(opt.model_rnn_size, opt.game_action_space_total))

    def get_params(self):
        return list(self.parameters())

    def reset_parameters(self):
        opt = self.opt
        self.messages_mlp.linear1.reset_parameters()  # type: ignore
        self.rnn.reset_parameters()
        self.agent_lookup.reset_parameters()
        self.state_lookup.reset_parameters()
        self.prev_action_lookup.reset_parameters()
        if self.prev_message_lookup:
            self.prev_message_lookup.reset_parameters()
        if opt.comm_enabled and opt.model_dial:
            self.messages_mlp.batchnorm1.reset_parameters()  # type: ignore
        self.outputs.linear1.reset_parameters()  # type: ignore
        self.outputs.linear2.reset_parameters()  # type: ignore
        for p in self.rnn.parameters():
            p.data.uniform_(*self.init_param_range)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, messages, hidden, prev_action):
        opt = self.opt

        bs, n_agents, obs_size = obs.shape
        obs = torch.reshape(obs, (-1, obs_size))
        if extras is not None:
            extras = torch.reshape(extras, (*obs.shape[:-1], *self.extras_shape))
            obs = torch.concat((obs, extras), dim=-1)

        prev_message = None
        if not opt.model_dial:
            if opt.model_action_aware:
                prev_action, prev_message = prev_action
                prev_action = prev_action.to(self.device)
                prev_message = prev_message.to(self.device)
                messages = messages.to(self.device)
        # agent_index = Variable(agent_index)

        z_a, z_o, z_u, z_m = [0] * 4
        # z_a = self.agent_lookup(agent_index)
        z_o = self.state_lookup(obs)
        if opt.model_action_aware:
            z_u = self.prev_action_lookup(prev_action)
            if prev_message is not None and self.prev_message_lookup is not None:
                z_u = z_u + self.prev_message_lookup(prev_message)

        z_u = z_u.reshape(bs * n_agents, -1)  # type: ignore

        z_m = self.messages_mlp(messages.view(-1, self.comm_size))
        z = z_a + z_o + z_u + z_m
        z = z.unsqueeze(1)

        # Reshape the hidden state to match the number of layers and batch size
        hidden_batch = hidden.view(opt.model_rnn_layers, bs * n_agents, -1)

        rnn_out, h_out = self.rnn(z, hidden_batch)
        outputs = self.outputs(rnn_out[:, -1, :].squeeze())

        return h_out.view(opt.model_rnn_layers, n_agents, bs, -1), outputs.view(bs, n_agents, -1)

    @classmethod
    def from_env(cls, env: MARLEnv, opt):
        assert len(env.observation_shape) == 1
        assert len(env.extra_shape) == 1
        return cls(env.observation_shape, env.extra_shape, opt.game_action_space_total, opt)
