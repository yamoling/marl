import numpy as np
import torch
from rlenv import RLEnv, Observation, Transition
from .policy import Policy
import marl



class DQN4:
    def __init__(self, env: RLEnv, policy: Policy, seed: int, gamma=0.99, batch_size=32, update_period=200):
        self.env = env
        self.device = torch.device(f"cuda:{seed % 3}" if torch.cuda.is_available() else "cpu")
        self.policy = policy
        self.dqn = marl.qlearning.MixedDQN(
            qnetwork=marl.nn.model_bank.MLP.from_env(env).to(self.device),
            mixer = marl.qlearning.mixers.VDN(env.n_agents),
            device=self.device,
            gamma=gamma,
            batch_size=batch_size,
            train_policy=self.policy,
            lr=1e-4,
            memory=marl.models.TransitionMemory(10_000)
        )
        self.update_step = 0
        self.target_update_period = update_period

    def choose_action(self, obs: Observation) -> int:
        return self.dqn.choose_action(obs)
    

    def update(self, transition: Transition):
        self.dqn.after_train_step(transition, self.update_step)
        self.update_step += 1
        # self.memory.add(transition)
        # if len(self.memory) < self.batch_size:
        #     return
        # self.update_step += 1
        # batch = self.memory.sample(self.batch_size).to(self.device).for_individual_learners()

        # qvalues = self.qnetwork.forward(batch.obs)
        # qvalues = torch.gather(qvalues, dim=-1, index=batch.actions).squeeze(-1)
        # with torch.no_grad():
        #     next_qvalues = self.qtarget.forward(batch.obs_)
        #     next_qvalues = torch.max(next_qvalues, dim=-1).values
        # target = batch.rewards + self.gamma * next_qvalues * (1 - batch.dones)
        
        # td_error = target - qvalues
        # loss = torch.mean(td_error ** 2)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # self.policy.update()
        # if self.update_step % self.target_update_period == 0:
        #     self.qtarget.load_state_dict(self.qnetwork.state_dict())


    def run_episode(self):
        finished = False
        obs = self.env.reset()
        score = 0
        episode_length = 0
        while not finished:
            action = self.choose_action(obs)
            next_obs, reward, done, truncated, info = self.env.step(action)
            self.update(Transition(obs, action, reward, done, info, next_obs))
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
