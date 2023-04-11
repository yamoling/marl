import marl
import rlenv
import torch
from copy import deepcopy


class CriticNetwork(torch.nn.Module):
    def __init__(self, input_dims: tuple[int], n_actions):
        super().__init__()
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_dims[0] + n_actions, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.nn(x)


class ValueNetwork(torch.nn.Module):
    def __init__(self, input_dims: tuple[int]):
        super().__init__()
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_dims[0], 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.nn(x)

class ActorNetwork(torch.nn.Module):
    def __init__(self, input_dims: tuple[int], n_actions, max_action: float):
        super().__init__()
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_dims[0], 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.mu = torch.nn.Linear(256, n_actions)
        self.std = torch.nn.Linear(256, n_actions)
        self.max_action = max_action

    def forward(self, x):
        x = self.nn(x)
        mu = self.mu(x)
        std = self.std(x)
        std = torch.clamp(std, min=1e-6, max=1.)
        return mu, std
    
    def sample_normal(self, obs, reparametrize=True):
        mu, std = self.forward(obs)
        normal = torch.distributions.Normal(mu, std)
        if reparametrize:
            actions = normal.rsample()
        else:
            actions = normal.sample()
        log_prob = normal.log_prob(actions) - torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = torch.sum(log_prob, 1, keepdim=True)
        action = torch.tanh(actions) * self.max_action
        return action, log_prob
    
class SACAgent:
    def __init__(self, env: rlenv.RLEnv, alpha, beta, gamma=0.99, batch_size=256, reward_scale=2, tau=0.005) -> None:
        self.gamma = gamma
        self.tau = tau
        self.scale = reward_scale
        self.memory = marl.models.TransitionMemory(1_000_000)
        
        self.critic_1 = CriticNetwork(env.observation_shape, env.n_actions)
        self.critic_2 = CriticNetwork(env.observation_shape, env.n_actions)
        self.actor = ActorNetwork(env.observation_shape, env.n_actions, max_action=1)
        self.value = ValueNetwork(env.observation_shape)
        self.target = deepcopy(self.value)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=alpha)
        self.optimizer_others = torch.optim.Adam(self.critic_1.parameters() + self.critic_2.parameters() + self.value.parameters() + self.target.parameters(), lr=beta)

    def choose_action(self, observation):
        with torch.no_grad():
            observation = torch.tensor([observation], dtype=torch.float32)
            action, _ = self.actor.sample_normal(observation)
            return action.detach().numpy()
        
    def remember(self, transition):
        self.memory.add(transition)

    def soft_update(self):
        for target_param, param in zip(self.target.parameters(), self.value.parameters()):
            new_value = (1-self.tau) * target_param.data + self.tau * param.data
            target_param.data.copy_(new_value, non_blocking=True)

    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = self.memory.sample(batch_size)

        value = self.value.forward(batch.obs)
        value_ = self.target.forward(batch.obs_) * (1 - batch.dones)
        
        actions, log_prob = self.actor.sample_normal(batch.obs, reparametrize=False)


if __name__ == "__main__":
    env = rlenv.make("Pendulum-v0")