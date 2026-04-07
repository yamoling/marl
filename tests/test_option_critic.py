import torch
from marlenv.catalog import DiscreteMockEnv

from marl.models.nn import NN, Actor, QNetwork
from marl.nn.model_bank.options import SimpleOptionCritic
from marl.training.option_critic import OptionCritic


class TinyActor(Actor):
    def __init__(self, obs_size: int, extras_size: int, n_actions: int):
        super().__init__()
        self.net = torch.nn.Linear(obs_size + extras_size, n_actions)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor):
        x = torch.cat((obs, extras), dim=-1)
        return self.net(x)

    def policy(self, obs: torch.Tensor, extras: torch.Tensor, available_actions: torch.Tensor):
        logits = self.forward(obs, extras)
        logits = self.mask(logits, available_actions)
        return torch.distributions.Categorical(logits=logits)

    def __hash__(self):
        return hash(self.name)


class TinyQOptions(QNetwork):
    def __init__(self, obs_size: int, extras_size: int, n_options: int):
        super().__init__(n_options)
        self.net = torch.nn.Linear(obs_size + extras_size, n_options)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        x = torch.cat((obs, extras), dim=-1)
        return self.net(x)

    def __hash__(self):
        return hash(self.name)


class TinyTermination(NN):
    def __init__(self, obs_size: int, extras_size: int, n_options: int):
        super().__init__()
        self.net = torch.nn.Linear(obs_size + extras_size, n_options)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor):
        x = torch.cat((obs, extras), dim=-1)
        return torch.sigmoid(self.net(x))

    def __hash__(self):
        return hash(self.name)


def _make_trainer(env: DiscreteMockEnv, n_options: int = 3) -> OptionCritic:
    obs_size = env.observation_shape[0]
    extras_size = env.extras_shape[0]
    n_actions = env.n_actions
    n_agents = env.n_agents

    option_critic = SimpleOptionCritic(
        policies=[TinyActor(obs_size, extras_size, n_actions) for _ in range(n_options)],
        q_options=TinyQOptions(obs_size, extras_size, n_options),
        options_termination=TinyTermination(obs_size, extras_size, n_options),
        n_agents=n_agents,
    )

    trainer = OptionCritic(
        option_critic=option_critic,
        lr=1e-3,
        memory_size=128,
        batch_size=8,
        update_interval=1,
        target_update_interval=10,
    )
    return trainer


def test_option_critic_action_and_head_shapes():
    env = DiscreteMockEnv()
    trainer = _make_trainer(env)
    agent = trainer.make_agent()

    obs, _ = env.reset()
    action = agent.choose_action(obs)
    assert action.shape == (env.n_agents,)
