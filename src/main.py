import rlenv
import marl
import torch
import laser_env
import os
from marl.utils.env_pool import EnvPool
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


def get_env_pools():
    train_envs, test_envs = [], []
    for file_path in os.listdir("maps/train_maps"):
        file_path = os.path.join("maps/train_maps", file_path)
        train_envs.append(laser_env.StaticLaserEnv(file_path, laser_env.ObservationType.FLATTENED))
    for file_path in os.listdir("maps/test_maps"):
        file_path = os.path.join("maps/test_maps", file_path)
        test_envs.append(laser_env.StaticLaserEnv(file_path, laser_env.ObservationType.FLATTENED))
    return EnvPool(train_envs), EnvPool(test_envs)

if __name__ == "__main__":
    logdir = "logs/random-pool"
    #logdir = "test"
    env, test_env = get_env_pools()
    env = rlenv.Builder(env).agent_id().time_limit(30).build()
    test_env = rlenv.Builder(test_env).agent_id().time_limit(30).build()
    
    # E-greedy decreasing from 1 to 0.1 over 200000 steps
    min_eps = 0.1
    decrease_amount = (1 - min_eps) / 200000
    train_policy = marl.policy.DecreasingEpsilonGreedy(1, decrease_amount, min_eps)
    test_policy = marl.policy.ArgMax()
    qnetwork = marl.nn.model_bank.MLP.from_env(env)
    algo = marl.qlearning.LinearVDN(
        qnetwork=qnetwork,
        gamma=0.95,
        train_policy=train_policy,
        test_policy=test_policy
    )

    from marl.utils.random_algo import RandomAgent
    algo = RandomAgent(env.n_actions, env.n_agents)
    experiment = Experiment.create(logdir, algo=algo, env=env, n_steps=300_000, test_interval=5000, test_env=test_env)
    for i in range(3):
        runner = experiment.create_runner(seed=i)
        runner.train(n_tests=5)
    