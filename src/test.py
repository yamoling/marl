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


def get_env_pools(obs_type=laser_env.ObservationType.FLATTENED):
    train_envs, test_envs = [], []
    for file_path in os.listdir("maps/7x7/train"):
        file_path = os.path.join("maps/7x7/train", file_path)
        train_envs.append(laser_env.StaticLaserEnv(file_path, obs_type))
    for file_path in os.listdir("maps/7x7/test"):
        file_path = os.path.join("maps/7x7/test", file_path)
        test_envs.append(laser_env.StaticLaserEnv(file_path, obs_type))
    return EnvPool(train_envs), EnvPool(test_envs)

if __name__ == "__main__":
    logdir = "logs/vdn-env-pool-5x5"
    # logdir = "test"
    from laser_env import ObservationType
    env, test_env = get_env_pools(ObservationType.FLATTENED)
    env = rlenv.Builder(env).agent_id().time_limit(45).build()
    test_env = rlenv.Builder(test_env).agent_id().time_limit(45).build()
    
    # E-greedy decreasing from 1 to 0.1 over 300000 steps
    min_eps = 0.1
    decrease_amount = (1 - min_eps) / 300000
    train_policy = marl.policy.DecreasingEpsilonGreedy(1, decrease_amount, min_eps)
    test_policy = marl.policy.ArgMax()
    qnetwork = marl.nn.model_bank.MLP.from_env(env)
    algo = marl.qlearning.LinearVDN(
        qnetwork=qnetwork,
        gamma=0.95,
        train_policy=train_policy,
        test_policy=test_policy
    )
    
    experiment = Experiment.create(logdir, algo=algo, env=env, n_steps=600_000, test_interval=5000, test_env=test_env)
    runner = experiment.create_runner(seed=0)
       
    