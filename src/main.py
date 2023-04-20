import rlenv
import marl
import torch
import laser_env
from marl import nn
from marl.models import Experiment


class ACNetwork(nn.ActorCriticNN):
    def __init__(self, input_shape: tuple, extras_shape: tuple | None, output_shape: tuple):
        super().__init__(input_shape, extras_shape, output_shape)
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


class ACNetwork2(nn.ActorCriticNN):
    def __init__(self, input_shape: tuple, extras_shape: tuple | None, output_shape: tuple):
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
    def value_parameters(self) -> list[torch.nn.Parameter]:
        return self.value_network.parameters()

    @property
    def policy_parameters(self) -> list[torch.nn.Parameter]:
        return self.policy_network.parameters()


nn.register(ACNetwork)
nn.register(ACNetwork2)


def load(logdir: str):
    experiment = Experiment.load(logdir)
    runner = experiment.create_runner()
    runner.train(n_tests=5)
    runner.train(n_tests=5)


if __name__ == "__main__":
    logdir = "logs/ppo"
    logdir="test-seed-3"
    env = rlenv.make("CartPole-v1")
    
    algo = marl.policy_gradient.PPO(
        lr_actor=3e-4, 
        lr_critic=1e-3 / 4, 
        gamma=0.99, 
        ac_network=ACNetwork2.from_env(env),
        c1=0.5 * 4
    )
    experiment = Experiment.create(logdir, algo=algo, env=env, n_steps=10_000, test_interval=1000)
    for i in range(2):
        runner = experiment.create_runner(seed=1)
        runner.train(n_tests=5)
