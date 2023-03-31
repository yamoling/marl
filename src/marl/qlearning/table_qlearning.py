import torch
import pickle
import numpy as np
from rlenv import Observation, Transition
from .qlearning import QLearning
from marl.models import TransitionMemory
from marl.policy import Policy, EpsilonGreedy, DecreasingEpsilonGreedy
from marl.utils import defaults_to


class VanillaQLearning(QLearning):
    def __init__(
        self, 
        train_policy: Policy=None,
        test_policy: Policy=None,
        lr=0.1,
        gamma=0.99,
    ):
        train_policy = defaults_to(train_policy, lambda: DecreasingEpsilonGreedy(decrease_amount=5e-4))
        test_policy = defaults_to(test_policy, lambda: EpsilonGreedy(0.01))
        super().__init__(train_policy, test_policy, gamma)
        self.lr = lr
        self.qtable: dict[int, np.ndarray[np.float32]] = {}

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

    def after_step(self, transition: Transition, time_step: int):
        qvalues = self.compute_qvalues(transition.obs).numpy()
        actions = transition.action[:, np.newaxis]
        qvalues = np.take_along_axis(qvalues, actions, axis=-1)

        next_qvalues = self.compute_qvalues(transition.obs_).numpy()
        next_qvalues = np.max(next_qvalues, axis=-1, keepdims=True)
        target_qvalues = transition.reward + self._gamma * next_qvalues

        new_qvalues = (1 - self.lr) * qvalues + self.lr * target_qvalues
        
        obs_data = np.concatenate((transition.obs.data, transition.obs.extras), axis=-1)
        for o, a, q in zip(obs_data, transition.action, new_qvalues):
            h = self.hash_ndarray(o)
            self.qtable[h][a] = q
        return super().after_step(transition, time_step)

    @staticmethod
    def hash_ndarray(data: np.ndarray) -> int:
        return hash(data.tobytes())

    def save(self, to_path: str):
        import os
        qtable_file = os.path.join(to_path, "qtable.pkl")
        train_policy_file = os.path.join(to_path, "train_policy")
        test_policy_file = os.path.join(to_path, "test_policy")
        self.train_policy.save(train_policy_file)
        self.test_policy.save(test_policy_file)
        with open(qtable_file, "wb") as f:
            pickle.dump(self.qtable, f)

    def load(self, from_path: str):
        import os
        qtable_file = os.path.join(from_path, "qtable.pkl")
        train_policy_file = os.path.join(from_path, "train_policy")
        test_policy_file = os.path.join(from_path, "test_policy")
        self.train_policy.load(train_policy_file)
        self.test_policy.load(test_policy_file)
        with open(qtable_file, "rb") as f:
            self.qtable = pickle.load(f)

    def summary(self) -> dict:
        return {
            **super().summary(),
            "train_policy": self.train_policy.__class__.__name__,
            "test_policy": self.test_policy.__class__.__name__,
            "gamma": self._gamma,
            "lr": self.lr
        }



class ReplayTableQLearning(VanillaQLearning):
    def __init__(
        self, 
        train_policy: Policy = None, 
        test_policy: Policy = None, 
        lr=0.1, 
        gamma=0.99, 
        replay_memory: TransitionMemory=None,
        batch_size=64
    ):
        super().__init__(train_policy, test_policy, lr, gamma)
        self.memory = defaults_to(replay_memory, lambda: TransitionMemory(50_000))
        self.batch_size = batch_size

    def batch_get(self, obs_data: np.ndarray, n_actions: int) -> np.ndarray[np.float32]:
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


    def after_step(self, transition: Transition, time_step: int):
        self.memory.add(transition)
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size).for_individual_learners()
        actions = batch.actions.numpy()

        obs_data = torch.concat([batch.obs, batch.extras], dim=-1).numpy()
        qvalues = self.batch_get(obs_data, transition.n_actions)
        qvalues = np.squeeze(np.take_along_axis(qvalues, actions, axis=-1), axis=-1)

        next_obs_data = torch.concat([batch.obs_, batch.extras_], dim=-1).numpy()
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
        self.memory.update(batch, qvalues, qtargets)

    def summary(self) -> dict:
        return {
            **super().summary(),
            "memory": self.memory.summary(),
            "lr": self.lr
        }
        