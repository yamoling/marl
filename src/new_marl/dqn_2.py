import numpy as np
import torch
import random
from rlenv import RLEnv, Observation
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


class DQN2:
    def __init__(self, env: RLEnv, policy: Policy, seed: int, gamma=0.99, batch_size=32, update_period=200):
        self.env = env
        self.device = torch.device(f"cuda:{seed % 3}" if torch.cuda.is_available() else "cpu")
        self.qnetwork = torch.nn.Sequential(
            torch.nn.Linear(*env.observation_shape, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, env.n_actions)
        ).to(self.device)
        self.qtarget = deepcopy(self.qnetwork).to(self.device)
        self.memory = deque(maxlen=10_000)
        self.gamma = gamma
        self.policy = policy
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.qnetwork.parameters(), lr=1e-4)
        self.update_step = 0
        self.target_update_period = update_period

    def choose_action(self, obs: Observation) -> int:
        obs = torch.from_numpy(obs).to(self.device, non_blocking=True)
        qvalues = self.qnetwork.forward(obs)
        return self.policy.choose_action(qvalues.unsqueeze(0).numpy(force=True), np.array([[1, 1]]))[0]
    

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
        obs = self.env.reset()
        score = 0
        episode_length = 0
        while not finished:
            action = self.choose_action(obs.data[0])
            next_obs, reward, done, truncated, info = self.env.step([action])
            self.update((obs.data[0], action, reward, done, next_obs.data[0]))
            finished = done or truncated
            score += reward
            episode_length += 1
            obs = next_obs
        return score, episode_length

    def train(self, n_steps: int):
        i = 0
        scores = []
        steps = []
        durations = []
        import time
        while i < n_steps:
            start = time.time()
            score, episode_length = self.run_episode()
            durations.append((time.time() - start)/episode_length)
            print(f"Step {i}\tscore: {score}\tavg_score: {np.mean(scores[-50:]):.3f}\tavg_duration: {np.mean(durations[-50:]):.5f}")
            i += episode_length
            scores.append(score)
            steps.append(i)
        return steps, scores
