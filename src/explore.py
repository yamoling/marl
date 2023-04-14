import rlenv
import marl
import torch
from marl import nn
from marl.models import Experiment


class ACNetwork(nn.ActorCriticNN):
    def __init__(self, input_shape: tuple, extras_shape: tuple|None, output_shape: tuple):
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
    

class ACNetwork2(nn.ActorCriticNN):
    def __init__(self, input_shape: tuple, extras_shape: tuple|None, output_shape: tuple):
        super().__init__(input_shape, extras_shape, output_shape)
        self.value_network = torch.nn.Sequential(
            torch.nn.Linear(*input_shape, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
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

nn.register(ACNetwork)
nn.register(ACNetwork2)

def load(logdir: str):
    experiment = Experiment.load(logdir)
    runner = experiment.create_runner()
    runner.train(n_tests=5)
    runner.train(n_tests=5)


if __name__ == "__main__":
    logdir = "logs/TD-A2C-one-network"
    env: rlenv.RLEnv[rlenv.models.DiscreteActionSpace] = rlenv.make("CartPole-v1")

    algo = marl.policy_gradient.TDActorCritic(
        alpha=5e-4,
        gamma=0.99,
        ac_network=ACNetwork.from_env(env)
    )
    experiment = Experiment.create(logdir, algo=algo, env=env, n_steps=300_000, test_interval=5000)
    for i in range(3):
        runner = experiment.create_runner()
        runner.train(n_tests=5)
