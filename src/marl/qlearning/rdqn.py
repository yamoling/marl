from dataclasses import dataclass
import torch
from rlenv import Observation, Episode
from marl import nn
from marl.models import EpisodeMemory, Batch
from marl.policy import Policy
from marl.utils import defaults_to

from .dqn import DQN

@dataclass
class RDQN(DQN):
    # Type hinting
    _qnetwork: nn.RecurrentNN
    _qtarget: nn.RecurrentNN
    _memory: EpisodeMemory
    _hidden_states: torch.Tensor | None

    def __init__(
        self, 
        qnetwork: nn.RecurrentNN, 
        gamma=0.99,
        tau=0.01, 
        batch_size=64,
        lr=1e-4,
        optimizer: torch.optim.Optimizer=None, 
        train_policy: Policy=None, 
        test_policy: Policy=None, 
        memory: EpisodeMemory=None, 
        device: torch.device=None,
    ) -> None:
        super().__init__(
            gamma=gamma, 
            tau=tau, 
            batch_size=batch_size,
            lr=lr, 
            optimizer=optimizer, 
            train_policy=train_policy, 
            test_policy=test_policy,
            device=device,
            memory=defaults_to(memory, lambda: EpisodeMemory(50_000)), 
            qnetwork=qnetwork
        )
        self._hidden_states=None
        

    def after_step(self, _time_step: int, _transition):
        # Override DQN behaviour: nothing to do after step in RDQN
        pass

    def after_train_episode(self, episode_num: int, episode: Episode):
        self._memory.add(episode)
        self.update()

    def before_train_episode(self, episode_num: int):
        self._hidden_states=None

    def _sample(self) -> Batch:
        return self._memory.sample(self._batch_size)\
            .for_rnn()\
            .for_individual_learners()

    def compute_qvalues(self, data: Batch | Observation) -> torch.Tensor:
        match data:
            case Batch() as batch:
                qvalues = self._qnetwork.forward(batch.obs, batch.extras)[0]
                qvalues = qvalues.view(batch.max_episode_len, batch.size, batch.n_agents, batch.n_actions)
                qvalues = qvalues.gather(index=batch.actions, dim=-1).squeeze(-1)
                return qvalues
            case Observation() as obs:
                obs_data = torch.from_numpy(obs.data).unsqueeze(0).to(self._device, non_blocking=True)
                obs_extras = torch.from_numpy(obs.extras).unsqueeze(0).to(self._device, non_blocking=True)
                qvalues, self._hidden_states = self._qnetwork.forward(obs_data, obs_extras, self._hidden_states)
                return qvalues.squeeze(0)
            case _: raise ValueError("Invalid input data type for 'compute_qvalues'")

    def compute_loss(self, qvalues: torch.Tensor, qtargets: torch.Tensor, batch: Batch) -> torch.Tensor:
        # Masked MSE loss (because some episodes have been padded)
        error = qtargets - qvalues
        masked_error = error * batch.masks
        criterion = torch.sum(masked_error ** 2, dim=0)
        loss = criterion.sum() / batch.masks.sum()
        return loss

    def compute_targets(self, batch: Batch) -> torch.Tensor:
        next_qvalues, _ = self._qtarget.forward(batch.obs_, batch.extras_)
        next_qvalues[batch.available_actions_ == 0.0] = -torch.inf
        next_qvalues: torch.Tensor = torch.max(next_qvalues, dim=-1)[0]
        next_qvalues = next_qvalues.view(batch.max_episode_len, batch.size, batch.n_agents)
        targets = batch.rewards + self._gamma * next_qvalues * (1 - batch.dones)
        return targets

    def summary(self) -> dict[str,]:
        summary = super().summary()
        summary["name"] = "RDQN"
        summary["recurrent"] = True
        return summary