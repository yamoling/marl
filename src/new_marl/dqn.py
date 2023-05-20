import numpy as np
import torch
import random
from gymnasium import Env
from copy import deepcopy
from dataclasses import dataclass
from collections import deque
from .policy import Policy

@dataclass
class Episode:
    obs: list[np.ndarray]
    actions: list[int]
    rewards: list[float]
    dones: list[bool]

Transition = tuple[np.ndarray, int, float, bool, np.ndarray]


class DQN:
    def __init__(self, env: Env, policy: Policy, seed: int, gamma=0.99, batch_size=32, update_period=200):
        self.env = env
        self.device = torch.device(f"cuda:{seed % 3}" if torch.cuda.is_available() else "cpu")
        self.qnetwork = torch.nn.Sequential(
            torch.nn.Linear(*env.observation_space.shape, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, env.action_space.n)
        ).to(self.device)
        self.qtarget = deepcopy(self.qnetwork).to(self.device)
        self.memory = deque(maxlen=10_000)
        self.gamma = gamma
        self.policy = policy
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.qnetwork.parameters(), lr=1e-4)
        self.update_step = 0
        self.target_update_period = update_period

    def choose_action(self, obs: np.ndarray) -> int:
        obs = torch.from_numpy(obs).to(self.device, non_blocking=True)
        qvalues = self.qnetwork.forward(obs)
        return self.policy.choose_action(qvalues)
    

    def update(self, transition: Transition):
        self.memory.append(transition)
        if len(self.memory) < self.batch_size:
            return
        self.update_step += 1
        batch = random.sample(self.memory, self.batch_size)
        obs, actions, rewards, dones, next_obs = zip(*batch)

        obs = torch.from_numpy(np.stack(obs, dtype=np.float32)).to(self.device, non_blocking=True)
        actions = torch.from_numpy(np.array(actions, dtype=np.int64)).unsqueeze(-1).to(self.device, non_blocking=True)
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32)).to(self.device, non_blocking=True)
        dones = torch.from_numpy(np.array(dones, dtype=np.float32)).to(self.device, non_blocking=True)
        next_obs = torch.from_numpy(np.stack(next_obs, dtype=np.float32)).to(self.device, non_blocking=True)

        qvalues = self.qnetwork.forward(obs)
        qvalues = torch.gather(qvalues, dim=-1, index=actions).squeeze(-1)
        with torch.no_grad():
            next_qvalues = self.qtarget.forward(next_obs)
            next_qvalues = torch.max(next_qvalues, dim=-1).values
        target = rewards + self.gamma * next_qvalues * (1 - dones)
        
        td_error = target - qvalues
        loss = torch.mean(td_error ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.policy.update()
        if self.update_step % self.target_update_period == 0:
            self.qtarget.load_state_dict(self.qnetwork.state_dict())


    def run_episode(self):
        finished = False
        obs, _ = self.env.reset()
        score = 0
        episode_length = 0
        while not finished:
            action = self.choose_action(obs)
            next_obs, reward, done, truncated, info = self.env.step(action)
            self.update((obs, action, reward, done, next_obs))
            finished = done or truncated
            score += reward
            episode_length += 1
            obs = next_obs
        return score, episode_length

    def train(self, n_steps: int):
        i = 0
        scores = []
        steps = []
        while i < n_steps:
            score, episode_length = self.run_episode()
            print(f"Step {i}\tscore: {score}\tavg_score: {np.mean(scores[-50:])}")
            i += episode_length
            scores.append(score)
            steps.append(i)
        return steps, scores
