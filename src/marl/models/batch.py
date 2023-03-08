from dataclasses import dataclass
from rlenv import Episode, Transition
import torch
import numpy as np


@dataclass
class Batch:
    """Batch model containing all the data required during training"""
    size: int
    max_episode_len: int
    n_agents: int
    n_actions: int
    obs: torch.Tensor
    extras: torch.Tensor
    obs_: torch.Tensor
    extras_: torch.Tensor
    actions: torch.LongTensor
    dones: torch.Tensor
    available_actions_: torch.LongTensor
    rewards: torch.Tensor
    masks: torch.Tensor
    states: torch.Tensor
    states_: torch.Tensor
    is_weights: torch.Tensor|None = None
    """Importance Sampling weights"""
    sample_index: list[int]|None = None

    @property
    def one_hot_actions(self) -> torch.Tensor:
        """One hot encoded actions"""
        one_hot: torch.Tensor = torch.nn.functional.one_hot(self.actions).to(torch.float32)
        one_hot = one_hot.squeeze(-2)
        return one_hot

    @staticmethod
    def from_episodes(episodes: list[Episode]) -> "Batch":
        """Create a batch from a list of episodes with shape (episode_len, batch_size, n_agents, ...)"""
        size = len(episodes)
        max_episode_len = max(len(e) for e in episodes)
        episodes = [e.padded(max_episode_len) for e in episodes]
        obs = torch.tensor(np.array([e.obs for e in episodes])).transpose(1, 0)
        extras = torch.tensor(np.array([e.extras for e in episodes])).transpose(1, 0)
        extras_ = torch.tensor(np.array([e.extras_ for e in episodes])).transpose(1, 0)
        obs_ = torch.tensor(np.array([e.obs_ for e in episodes])).transpose(1, 0)
        actions = torch.tensor(np.array([e.actions for e in episodes])).transpose(1, 0).unsqueeze(-1)
        dones = torch.tensor(np.array([e.dones for e in episodes])).transpose(1, 0)
        available_actions_ = torch.tensor(np.array([e.available_actions_ for e in episodes])).transpose(1, 0)
        rewards = torch.tensor(np.array([e.rewards for e in episodes]), dtype=torch.float32).transpose(1, 0)
        masks = torch.tensor(np.array([e.mask for e in episodes])).transpose(1, 0)
        states = torch.tensor(np.array([e.states[:-1] for e in episodes])).transpose(1, 0)
        states_ = torch.tensor(np.array([e.states[1:] for e in episodes])).transpose(1, 0)
        _, _, n_agents, n_actions = available_actions_.shape[-1]
        return Batch(
            size=size,
            max_episode_len=max_episode_len,
            n_agents=n_agents,
            n_actions=n_actions,
            obs=obs,
            extras=extras,
            obs_=obs_,
            extras_=extras_,
            actions=actions,
            dones=dones,
            available_actions_=available_actions_,
            rewards=rewards,
            masks=masks,
            states=states,
            states_=states_
        )

    @staticmethod
    def from_transitions(transitions: list[Transition]) ->"Batch":
        """Create a batch from a list of transitions"""
        obs, obs_, extras, extras_, dones, actions, available_actions_, rewards, states, states_ = [], [], [], [], [], [], [], [], [], []
        for t in transitions:
            obs.append(t.obs.data)
            extras.append(t.obs.extras)
            obs_.append(t.obs_.data)
            extras_.append(t.obs_.extras)
            dones.append(t.done)
            actions.append(t.action)
            available_actions_.append(t.obs_.available_actions)
            rewards.append(t.reward)
            states.append(t.obs.state)
            states_.append(t.obs_.state)
        batch_size = len(transitions)
        n_agents = transitions[0].n_agents
        n_actions = transitions[0].obs.available_actions.shape[-1]
        return Batch(
            size=batch_size,
            max_episode_len=1,
            n_agents=n_agents,
            n_actions=n_actions,
            obs=torch.from_numpy(np.array(obs, dtype=np.float32)),
            extras=torch.from_numpy(np.array(extras, dtype=np.float32)),
            obs_=torch.from_numpy(np.array(obs_, dtype=np.float32)),
            extras_=torch.from_numpy(np.array(extras_, dtype=np.float32)),
            actions=torch.from_numpy(np.array(actions, dtype=np.int64)).unsqueeze(-1),
            dones=torch.from_numpy(np.array(dones, dtype=np.float32)),
            available_actions_=torch.from_numpy(np.array(available_actions_, dtype=np.int64)),
            rewards=torch.from_numpy(np.array(rewards, dtype=np.float32)),
            masks=torch.ones(batch_size, dtype=torch.float32),
            states=torch.from_numpy(np.array(states, dtype=np.float32)),
            states_=torch.from_numpy(np.array(states_, dtype=np.float32))
        )

    @staticmethod
    def from_transition_slices(transitions: list[list[Transition]]) -> "Batch":
        n_agents = transitions[0][0].n_agents
        n_actions = transitions[0][0].obs.available_actions.shape[-1]
        size = len(transitions)
        slice_length = len(transitions[0])
        obs, extras, actions, rewards, dones, obs_, extras_, available_actions_, states, states_, masks = [], [], [], [], [], [], [], [], [], [], []
        for slices in transitions:
            obs.append([t.obs.data for t in slices])
            extras.append([t.obs.extras for t in slices])
            actions.append([t.action for t in slices])
            rewards.append([t.reward for t in slices])
            dones.append([t.done for t in slices])
            obs_.append([t.obs_.data for t in slices])
            extras_.append([t.obs_.extras for t in slices])
            available_actions_.append([t.obs_.available_actions for t in slices])
            states.append([t.obs.state for t in slices])
            states_.append([t.obs_.state for t in slices])
            # Mask transitions that are not from the same episode as **the first one**
            # - transitions of the same episode get 1.0
            # - transitions of an other episode get 0.0
            done = False
            mask = []
            for t in slices:
                mask.append(float(not done))
                done = done or t.is_done
            masks.append(mask)
        masks = np.array(masks, dtype=np.float32)
        
        return Batch(
            size=size,
            max_episode_len=slice_length,
            n_agents=n_agents,
            n_actions=n_actions,
            obs=torch.from_numpy(np.array(obs)),
            obs_ = torch.from_numpy(np.array(obs_)),
            extras=torch.from_numpy(np.array(extras)),
            extras_=torch.from_numpy(np.array(extras_)),
            actions=torch.from_numpy(np.array(actions)).unsqueeze(-1),
            available_actions_=torch.from_numpy(np.array(available_actions_)),
            rewards=torch.from_numpy(np.array(rewards, dtype=np.float32)),
            states=torch.from_numpy(np.array(states)),
            states_=torch.from_numpy(np.array(states_)),
            dones=torch.from_numpy(np.array(dones)),
            masks=torch.from_numpy(np.array(masks, dtype=np.float32))
        )

    def for_individual_learners(self) -> "Batch":
        """Reshape rewards, dones and masks such that each agent has its own (identical) signal."""
        self.rewards = self.rewards.repeat_interleave(self.n_agents).reshape(*self.rewards.shape, self.n_agents)
        self.dones = self.dones.repeat_interleave(self.n_agents).reshape(*self.dones.shape, self.n_agents)
        self.masks = self.masks.repeat_interleave(self.n_agents).reshape(*self.masks.shape, self.n_agents)
        return self

    def for_rnn(self) -> "Batch":
        """Reshape observations, extras and available actions such that dimensions 1 and 2 are merged (required for GRU)"""
        obs_size = self.obs.shape[3:]
        self.obs = self.obs.reshape(self.max_episode_len, self.size * self.n_agents, *obs_size)
        self.obs_ = self.obs_.reshape(self.max_episode_len, self.size * self.n_agents, *obs_size)
        self.extras = self.extras.reshape(self.max_episode_len, self.size * self.n_agents, -1)
        self.extras_ = self.extras_.reshape(self.max_episode_len, self.size * self.n_agents, -1)
        self.available_actions_ = self.available_actions_.reshape(self.max_episode_len, self.size * self.n_agents, self.n_actions)
        return self

    def to(self, device: torch.device) -> "Batch":
        """Send the tensors to the given device"""
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                value = value.to(device, non_blocking=True)
                setattr(self, key, value)
        return self

    @property
    def device(self) -> torch.device:
        return self.obs.device
