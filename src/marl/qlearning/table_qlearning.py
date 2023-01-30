import numpy as np
from rlenv import RLEnv, Observation, Transition
from marl.marl_algo import RLAlgorithm
from marl.policy import Policy, EpsilonGreedy, DecreasingEpsilonGreedy
from marl.utils import defaults_to

class TableQLearning(RLAlgorithm):
    def __init__(
        self, 
        env: RLEnv, 
        test_env: RLEnv, 
        train_policy: Policy=None,
        test_policy: Policy=None,
        lr=0.1,
        gamma=0.99,
        log_path: str = None
    ):
        super().__init__(env, test_env, log_path)
        self.qtable: dict[int, np.ndarray[np.float32]] = {}
        self.train_policy = defaults_to(train_policy, DecreasingEpsilonGreedy(env.n_agents, decrease_amount=5e-4))
        self.test_policy = defaults_to(test_policy, EpsilonGreedy(env.n_agents, 0.01))
        self.policy = train_policy
        self.lr = lr
        self.gamma = gamma

    def get_qvalues(self, obs: Observation) -> np.ndarray[np.float32]:
        qvalues = []
        obs_data = np.concatenate((obs.data, obs.extras), axis=-1)
        for agent_obs in obs_data:
            h = hash(agent_obs.tobytes())
            if h not in self.qtable:
                self.qtable[h] = np.ones(self.env.n_actions, dtype=np.float32)
            agent_qvalues = self.qtable[h]
            qvalues.append(agent_qvalues)            
        return np.array(qvalues)

    def choose_action(self, observation: Observation) -> np.ndarray[np.int64]:
        qvalues = self.get_qvalues(observation)
        return self.policy.get_action(qvalues, observation.available_actions)

    def after_step(self, transition: Transition, step_num: int):
        qvalues = self.get_qvalues(transition.obs)
        actions = transition.action[:, np.newaxis]
        qvalues = np.take_along_axis(qvalues, actions, axis=-1)

        next_qvalues = self.get_qvalues(transition.obs_)
        next_qvalues = np.max(next_qvalues, axis=-1, keepdims=True)
        target_qvalues = transition.reward + self.gamma * next_qvalues

        new_qvalues = (1 - self.lr) * qvalues + self.lr * target_qvalues
        
        obs_data = np.concatenate((transition.obs.data, transition.obs.extras), axis=-1)
        for o, a, q in zip(obs_data, transition.action, new_qvalues):
            h = hash(o.tobytes())
            self.qtable[h][a] = q
        return super().after_step(transition, step_num)

    def before_tests(self):
        self.policy = self.test_policy
        return super().before_tests()

    def after_tests(self, time_step: int, episodes):
        self.policy = self.train_policy
        return super().after_tests(time_step, episodes)

    def save(self, to_path: str):
        return