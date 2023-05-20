import numpy as np
import torch
from rlenv.models import RLEnv, Observation, Transition, Episode, EpisodeBuilder
from copy import deepcopy
import marl
from marl.nn import RecurrentNN
from marl import Batch

from .policy import Policy
from .mixers import Mixer, VDN


class RecurrentMixedDQN:
    def __init__(
            self, 
            env: RLEnv,
            policy: Policy,
            seed: int,
            qnetwork: RecurrentNN,
            memory: marl.models.EpisodeMemory,
            mixer: Mixer=None,
            gamma=0.99,
            batch_size=32,
            update_period=100,
            ddqn=True
        ):
        self.env = env
        if mixer is None:
            mixer = VDN()
        self.mixer = mixer
        self.target_mixer = deepcopy(mixer)
        self.device = torch.device(f"cuda:{seed % 3}" if torch.cuda.is_available() else "cpu")
        self.qnetwork = qnetwork.to(self.device)
        self.qtarget = deepcopy(self.qnetwork).to(self.device)
        self.memory = memory
        self.gamma = gamma
        self.policy = policy
        self.batch_size = batch_size
        self._parameters = list(self.qnetwork.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(self._parameters, lr=5e-3)
        self.target_update_period = update_period
        self.double_qlearning = ddqn

        self.update_step = 0
        self.hidden_state = None

    @torch.no_grad()
    def choose_action(self, obs: Observation) -> list[int]:
        obs_data = torch.from_numpy(obs.data).to(self.device, non_blocking=True)
        extras = torch.from_numpy(obs.extras).to(self.device, non_blocking=True)
        qvalues, self.hidden_state = self.qnetwork.forward(obs_data, extras, self.hidden_state)
        return self.policy.choose_action(qvalues.numpy(force=True), obs.available_actions)
    
    @torch.no_grad()
    def compute_targets(self, batch: Batch) -> torch.Tensor:
        # Add the first observation to the batch
        # obs_ = torch.concat([batch.obs[0].unsqueeze(0), batch.obs_])
        # extras_ = torch.concat([batch.extras[0].unsqueeze(0), batch.extras_])
        # target_next_qvalues = self.qtarget.forward(obs_, extras_)[0]
        # target_next_qvalues = target_next_qvalues[1:]
        target_next_qvalues = self.qtarget.forward(batch.obs_, batch.extras_)[0]
        if self.double_qlearning:
            # current_next_qvalues = self.qnetwork.forward(obs_, extras_)[0]
            current_next_qvalues = self.qnetwork.forward(batch.obs_, batch.extras_)[0]
            # current_next_qvalues = current_next_qvalues[1:]
            current_next_qvalues[batch.available_actions_ == 0.0] = -torch.inf
            indices = torch.argmax(current_next_qvalues, dim=-1, keepdim=True)
        else:
            target_next_qvalues[batch.available_actions_ == 0] = -torch.inf
            indices = torch.argmax(target_next_qvalues, dim=-1, keepdim=True)
        next_values = torch.gather(target_next_qvalues, dim=-1, index=indices).squeeze(-1)
        next_values = self.target_mixer.forward(next_values, batch.states_)
        targets = batch.rewards + self.gamma * next_values * (1 - batch.dones)
        return targets

    def update_transition(self, transition: Transition):
        self.policy.update()

    def update_episode(self, episode: Episode):
        self.memory.add(episode)
        if len(self.memory) < self.batch_size:
            return
        self.update_step += 1
        batch = self.memory.sample(self.batch_size).to(self.device)

        qvalues = self.qnetwork.forward(batch.obs, batch.extras)[0]
        qvalues = torch.gather(qvalues, dim=-1, index=batch.actions).squeeze(-1)
        qvalues = self.mixer.forward(qvalues, batch.states)
        target = self.compute_targets(batch)
        
        td_error = (target - qvalues) * batch.masks
        loss = torch.sum(td_error ** 2) / torch.sum(batch.masks)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self._parameters, 10)
        self.optimizer.step()

        if self.update_step % self.target_update_period == 0:
            self.qtarget.load_state_dict(self.qnetwork.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def run_episode(self):
        self.hidden_state = None
        finished = False
        obs = self.env.reset()
        score = 0
        episode_length = 0
        episode = EpisodeBuilder()
        while not finished:
            action = self.choose_action(obs)
            next_obs, reward, done, truncated, info = self.env.step(action)
            transition = Transition(obs, action, reward, done, info, next_obs)
            episode.add(transition)
            self.update_transition(transition)
            finished = done or truncated
            score += reward
            episode_length += 1
            obs = next_obs
        self.update_episode(episode.build())
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
