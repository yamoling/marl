import numpy as np
import torch
from rlenv.models import RLEnv, Observation, Episode, EpisodeBuilder, Transition
from copy import deepcopy
from .policy import Policy
from marl.nn.model_bank import RNNQMix
from marl.models.replay_memory import EpisodeMemory


class RDQN:
    def __init__(self, env: RLEnv, policy: Policy, seed: int, gamma=0.99, batch_size=32, update_period=200):
        self.env = env
        self.device = torch.device(f"cuda:{seed % 3}" if torch.cuda.is_available() else "cpu")
        self.qnetwork = RNNQMix.from_env(env).to(self.device)
        self.qtarget = deepcopy(self.qnetwork).to(self.device)
        self.memory = EpisodeMemory(5_000)
        self.gamma = gamma
        self.policy = policy
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.qnetwork.parameters(), lr=1e-3)
        self.update_step = 0
        self.target_update_period = update_period
        self.hidden_state = None

    def choose_action(self, obs: Observation) -> int:
        obs_data = torch.from_numpy(obs.data).to(self.device, non_blocking=True)
        qvalues, self.hidden_state = self.qnetwork.forward(obs_data, hidden_states=self.hidden_state)
        return self.policy.choose_action(qvalues.numpy(force=True), obs.available_actions)


    def episode_update(self, episode: Episode):
        self.memory.add(episode)
        if len(self.memory) < self.batch_size:
            return
        self.update_step += 1
        batch = self.memory.sample(self.batch_size).to(self.device)

        qvalues = self.qnetwork.forward(batch.obs)[0]
        qvalues = torch.gather(qvalues, dim=-1, index=batch.actions).squeeze()
        with torch.no_grad():
            next_qvalues = self.qtarget.forward(batch.obs_)[0]
            next_qvalues = torch.max(next_qvalues, dim=-1).values
        target = batch.rewards + self.gamma * next_qvalues.squeeze(-1) * (1 - batch.dones)
        
        td_error = (target - qvalues) * batch.masks
        loss = torch.sum(td_error ** 2) / torch.sum(batch.masks)
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
        self.hidden_state = None
        episode = EpisodeBuilder()
        while not finished:
            action = self.choose_action(obs)
            next_obs, reward, done, truncated, info = self.env.step(action)
            episode.add(Transition(obs, action, reward, done, info, next_obs))
            self.policy.update()
            finished = done or truncated
            score += reward
            episode_length += 1
            obs = next_obs
        self.episode_update(episode.build())
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
