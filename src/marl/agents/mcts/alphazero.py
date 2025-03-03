import numpy as np
import torch
from marlenv import MARLEnv, State, DiscreteActionSpace
from marl.nn.model_bank import CNN_ActorCritic
from collections import deque
from marl.logging import CSVLogger
from .alpha_node import AlphaNode
from datetime import datetime
from copy import deepcopy


class AlphaZero:
    def __init__(
        self,
        env: MARLEnv[list[int], DiscreteActionSpace],
        n_search_iterations: int = 100,
        lr: float = 0.001,
        exploration_constant: float = 2**0.5,
        gamma: float = 0.99,
        tau: float = 1.0,
        batch_size: int = 64,
    ):
        self.env = env
        self.gamma = gamma
        assert len(env.state_shape) == 3
        assert len(env.state_extra_shape) == 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = CNN_ActorCritic(
            env.state_shape,
            env.state_extra_shape,
            (env.action_space.n_actions,),
        ).to(self.device)
        self.tau = tau
        self.c = exploration_constant
        self.n_search_iterations = n_search_iterations
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.batch_size = batch_size

    def search(self, state: State):
        s = torch.from_numpy(state.data).unsqueeze(0).to(self.device)
        e = torch.from_numpy(state.extras).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.network.value(s, e).item()
        root = AlphaNode.root(state, value)
        for _ in range(self.n_search_iterations):
            self.update(root)
        return root

    def update(self, root: AlphaNode):
        while root.is_expanded:
            root = root.get_max_ucb_child(self.c)
        root.expand(self.env, self.network, self.gamma)
        root.backprop(root.value)

    def self_play(self, render: bool):
        env = deepcopy(self.env)
        state = env.reset()[1]
        states = [state]
        qvalues = []
        target_probs = []
        availables = []
        actions = []
        episode_return = 0.0
        done = False
        while not done:
            availables.append(env.available_actions()[0])
            root = self.search(state)
            child = root.get_child(self.tau)
            state = child.state
            qvalues.append(child.q_value)
            target_probs.append(child.target_prob(self.tau))
            states.append(child.state)
            actions.append(child.action)
            step = env.step([child.action])
            done = step.is_terminal
            episode_return += step.reward.item()
            if render:
                env.render()
        # Remove the last state which is terminal
        states.pop(-1)
        return states, qvalues, target_probs, availables, actions, episode_return

    def train_network(
        self,
        all_states: deque[State],
        all_qvalues: deque[float],
        all_target_probs: deque[float],
        all_actions: deque[int],
        all_availables: deque[np.ndarray],
    ):
        indices = np.random.choice(np.arange(len(all_states)), self.batch_size)

        states_data = torch.from_numpy(np.array([all_states[i].data for i in indices])).to(self.device)
        state_extras = torch.from_numpy(np.array([all_states[i].extras for i in indices])).to(self.device)
        qvalues = torch.from_numpy(np.array([all_qvalues[i] for i in indices], dtype=np.float32)).to(self.device)
        target_probs = torch.from_numpy(np.array([all_target_probs[i] for i in indices], dtype=np.float32)).to(self.device)
        actions = torch.from_numpy(np.array([all_actions[i] for i in indices])).to(self.device)
        availables = torch.from_numpy(np.array([all_availables[i] for i in indices])).to(self.device)

        pi, value = self.network.forward(states_data, state_extras, availables)
        pi = torch.gather(pi, 1, actions.unsqueeze(-1)).squeeze()
        value = value.squeeze()
        value_loss = torch.nn.functional.mse_loss(value, qvalues)
        policy_loss = torch.nn.functional.cross_entropy(pi, target_probs)
        loss = value_loss + policy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, n_epochs: int, buffer_size: int = 50_000):
        logger = CSVLogger(f"{datetime.now()}-alphazero-{self.env.name}.csv")
        all_states = deque(maxlen=buffer_size)
        all_qvalues = deque(maxlen=buffer_size)
        all_target_probs = deque(maxlen=buffer_size)
        all_actions = deque(maxlen=buffer_size)
        all_availables = deque(maxlen=buffer_size)
        for t in range(n_epochs):
            states, qvalues, target_probs, availables, actions, total = self.self_play(render=False)
            all_states.extend(states)
            all_qvalues.extend(qvalues)
            all_target_probs.extend(target_probs)
            all_actions.extend(actions)
            all_availables.extend(availables)
            loss = self.train_network(all_states, all_qvalues, all_target_probs, all_actions, all_availables)
            logger.log_print({"loss": loss, "total": total}, t)
