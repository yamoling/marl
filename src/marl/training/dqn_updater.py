from rlenv import Transition
import torch
from copy import deepcopy

from marl.models import Trainer, TransitionMemory
from marl.policy import Policy
from marl.nn import LinearNN


class DQNUpdater(Trainer):
    def __init__(self, qnetwork: LinearNN, train_policy: Policy, memory: TransitionMemory, lr: float):
        super().__init__(update_type="step", update_interval=1)
        self.qnetwork = qnetwork
        self.train_policy = train_policy
        self.device = torch.device("cpu")
        self.qtarget = deepcopy(self.qnetwork).to(self.device)
        self.optim = torch.optim.Adam(self.qnetwork.parameters(), lr=lr)
        self.memory = memory
        self.batch_size = 64
        self.gamma = 0.99

    
    def update_step(self, transition: Transition, time_step: int):
        self.memory.add(transition)
        if len(self.memory) < self.batch_size:
            return
        self.train_policy.update(time_step)

        import numpy as np
        indices = np.random.randint(0, len(self.memory), self.batch_size)
        transitions = [self.memory._memory[i] for i in indices]

        obs = torch.tensor(np.array([t.obs.data for t in transitions]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array([t.action for t in transitions]), dtype=torch.int64).to(self.device).unsqueeze(-1)
        rewards = torch.tensor(np.array([t.reward for t in transitions]), dtype=torch.float32).to(self.device)
        obs_ = torch.tensor(np.array([t.obs_.data for t in transitions]), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array([t.done for t in transitions]), dtype=torch.bool).to(self.device)

        batch = self.memory.get_batch(indices).to(self.device)
        assert torch.equal(batch.obs, obs)
        assert torch.equal(batch.actions, actions)
        assert torch.equal(batch.rewards, rewards)
        assert torch.equal(batch.obs_, obs_)
        assert torch.equal(batch.dones, dones)

        self.optim.zero_grad()
        qvalues = self.qnetwork.forward(obs)
        qvalues = torch.gather(qvalues, index=actions, dim=-1).squeeze()
        with torch.no_grad():
            next_qvalues = self.qtarget.forward(obs_)
            next_qvalues = torch.max(next_qvalues, dim=-1)[0]
            next_qvalues[dones] = 0.0
        qtargets = batch.rewards + self.gamma * torch.squeeze(next_qvalues)

        loss = torch.nn.functional.mse_loss(qvalues, qtargets)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.qnetwork.parameters(), 5.0)
        self.optim.step()
        if time_step % 200 == 0:
            self.qtarget.load_state_dict(self.qnetwork.state_dict())
            print(f"[{time_step:5d}]Loss: {loss.item()}, epsilon: {self.train_policy.epsilon.value}")



    def save(self, to_directory: str):
        return
    
    def load(self, from_directory: str):
        return

    def to(self, device: torch.device):
        self.device = device
        self.qnetwork = self.qnetwork.to(device)
        self.qtarget = self.qtarget.to(device)
        return self
    
    def randomize(self):
        self.qnetwork.randomize()
        self.qnetwork.randomize()