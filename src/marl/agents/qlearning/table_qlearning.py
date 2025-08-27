import torch
import pickle
from typing import Optional
import numpy as np
from marlenv import Episode, Observation, Transition
from marl.models import TransitionMemory, Policy, Trainer
from marl.agents import Agent


class VanillaQLearning(Agent, Trainer):
    def __init__(
        self,
        train_policy: Policy,
        test_policy: Optional[Policy] = None,
        lr=0.1,
        gamma=0.99,
    ):
        self._train_policy = train_policy
        if test_policy is None:
            test_policy = train_policy
        self._test_policy = test_policy
        self.train_policy = train_policy
        self.test_policy = test_policy
        self.policy = train_policy
        self._gamma = gamma
        self.lr = lr
        self.qtable: dict[int, np.ndarray] = {}

    def compute_qvalues(self, obs: Observation):
        qvalues = []
        obs_data = np.concatenate((obs.data, obs.extras), axis=-1)
        for agent_obs in obs_data:
            h = self.hash_ndarray(agent_obs)
            if h not in self.qtable:
                self.qtable[h] = np.ones(obs.available_actions.shape[-1], dtype=np.float32)
            agent_qvalues = self.qtable[h]
            qvalues.append(agent_qvalues)
        return torch.from_numpy(np.array(qvalues))

    def update_step(self, transition: Transition, time_step: int):
        qvalues = self.compute_qvalues(transition.obs).numpy()
        actions = transition.action[:, np.newaxis]
        qvalues = np.take_along_axis(qvalues, actions, axis=-1)

        next_qvalues = self.compute_qvalues(transition.next_obs).numpy()
        next_qvalues = np.max(next_qvalues, axis=-1, keepdims=True)
        target_qvalues = transition.reward + self._gamma * next_qvalues

        new_qvalues = (1 - self.lr) * qvalues + self.lr * target_qvalues

        obs_data = np.concatenate((transition.obs.data, transition.obs.extras), axis=-1)
        for o, a, q in zip(obs_data, transition.action, new_qvalues):
            h = self.hash_ndarray(o)
            self.qtable[h][a] = q
        raise NotImplementedError()

    @staticmethod
    def hash_ndarray(data: np.ndarray) -> int:
        return hash(data.tobytes())

    def save(self, to_path: str):
        import os

        qtable_file = os.path.join(to_path, "qlearning.pkl")
        with open(qtable_file, "wb") as f:
            pickle.dump(self, f)

    def load(self, from_path: str):
        import os

        file = os.path.join(from_path, "qtable.pkl")
        with open(file, "rb") as f:
            loaded: VanillaQLearning = pickle.load(f)
            self.qtable = loaded.qtable
            self.train_policy = loaded.train_policy
            self.test_policy = loaded.test_policy


class ReplayTableQLearning(VanillaQLearning):
    def __init__(
        self,
        train_policy: Policy,
        replay_memory: TransitionMemory,
        test_policy: Optional[Policy] = None,
        lr=0.1,
        gamma=0.99,
        batch_size=64,
    ):
        super().__init__(train_policy, test_policy, lr, gamma)
        self.memory = replay_memory
        self.batch_size = batch_size

    def batch_get(self, obs_data: np.ndarray, n_actions: int) -> np.ndarray:
        qvalues = []
        for obs in obs_data:
            agent_qvalues = []
            for agent_obs in obs:
                h = self.hash_ndarray(agent_obs)
                if h not in self.qtable:
                    self.qtable[h] = np.ones(n_actions, dtype=np.float32)
                agent_qvalues.append(self.qtable[h])
            qvalues.append(agent_qvalues)
        return np.array(qvalues)

    def after_train_step(self, transition: Transition, time_step: int):
        self.memory.add(transition)
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size).for_individual_learners()
        actions = batch.actions.numpy()

        obs_data = torch.concat([batch.obs, batch.extras], dim=-1).numpy()
        qvalues = self.batch_get(obs_data, transition.n_actions)
        qvalues = np.squeeze(np.take_along_axis(qvalues, actions, axis=-1), axis=-1)

        next_obs_data = torch.concat([batch.next_states, batch.next_extras], dim=-1).numpy()
        next_qvalues = self.batch_get(next_obs_data, transition.n_actions)
        next_qvalues = np.max(next_qvalues, axis=-1)
        qtargets = batch.rewards.numpy() + self._gamma * next_qvalues

        new_qvalues = (1 - self.lr) * qvalues + self.lr * qtargets

        # Setting new qvalues
        actions = np.squeeze(actions, axis=-1)
        for o, a, q in zip(obs_data, actions, new_qvalues):
            for agent_obs, agent_action, agent_q in zip(o, a, q):
                h = self.hash_ndarray(agent_obs)
                self.qtable[h][agent_action] = agent_q
