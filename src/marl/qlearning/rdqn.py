from dataclasses import dataclass
import torch
from rlenv import RLEnv, Observation, Episode
from marl import nn
from marl.models import EpisodeMemory, Batch
from marl.policy import Policy
from marl.utils import defaults_to

from .dqn import DQN

@dataclass
class RDQN(DQN):
    # Type hinting
    qnetwork: nn.RecurrentNN
    qtarget: nn.RecurrentNN
    memory: EpisodeMemory
    hidden_states: torch.Tensor | None

    def __init__(
        self, 
        env: RLEnv,
        test_env: RLEnv=None,
        gamma=0.99,
        tau=0.01, 
        batch_size=64,
        lr=1e-4,
        qnetwork: nn.RecurrentNN=None, 
        optimizer: torch.optim.Optimizer=None, 
        train_policy: Policy=None, 
        test_policy: Policy=None, 
        memory: EpisodeMemory=None, 
        device: torch.device=None,
        log_path: str=None
    ) -> None:
        super().__init__(
            env=env, 
            test_env=test_env,
            gamma=gamma, 
            tau=tau, 
            batch_size=batch_size,
            lr=lr, 
            optimizer=optimizer, 
            train_policy=train_policy, 
            test_policy=test_policy,
            device=device,
            memory=defaults_to(memory, EpisodeMemory(50_000)), 
            qnetwork=defaults_to(qnetwork, nn.model_bank.RNNQMix.from_env(env)),
            log_path=log_path
        )
        self.hidden_states=None
        

    def after_step(self, _step_num: int, _transition):
        """Override DQN behaviour: nothing to do after step in RDQN"""
        pass

    def after_episode(self, episode_num: int, episode: Episode):
        self.memory.add(episode)
        self.update()

    def before_episode(self, episode_num: int):
        self.hidden_states=None

    def _sample(self) -> Batch:
        return self.memory.sample(self.batch_size)\
            .for_rnn()\
            .for_individual_learners()

    def compute_qvalues(self, data: Batch | Observation) -> torch.Tensor:
        match data:
            case Batch() as batch:
                qvalues = self.qnetwork.forward(batch.obs, batch.extras)[0]
                qvalues = qvalues.reshape(batch.max_episode_len, batch.size, self.env.n_agents, self.env.n_actions)
                qvalues = qvalues.gather(index=batch.actions, dim=-1).squeeze(-1)
                return qvalues
            case Observation() as obs:
                obs_data = torch.from_numpy(obs.data).unsqueeze(0).to(self.device, non_blocking=True)
                obs_extras = torch.from_numpy(obs.extras).unsqueeze(0).to(self.device, non_blocking=True)
                qvalues, self.hidden_states = self.qnetwork.forward(obs_data, obs_extras, self.hidden_states)
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
        next_qvalues, _ = self.qtarget.forward(batch.obs_, batch.extras_)
        next_qvalues[batch.available_actions_ == 0.0] = -torch.inf
        next_qvalues: torch.Tensor = torch.max(next_qvalues, dim=-1)[0]
        next_qvalues = next_qvalues.reshape(batch.max_episode_len, batch.size, self.env.n_agents)
        targets = batch.rewards + self.gamma * next_qvalues * (1 - batch.dones)
        return targets

    def summary(self) -> dict[str,]:
        summary = super().summary()
        summary["name"] = "Recurrent DQN"
        return summary