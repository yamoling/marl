from rlenv.models import Episode
import torch
from rlenv import Observation
from marl.models import TransitionsBatch, EpisodeBatch, EpisodeMemory
from marl.nn import LinearNN, RecurrentNN
from marl import intrinsic_reward as ir
from marl.logging import Logger
from marl.policy import Policy
from marl.utils import defaults_to, get_device
from copy import deepcopy

from .dqn import DQN
from .mixers import Mixer

class LinearMixedDQN(DQN):
    def __init__(
            self, 
            qnetwork: LinearNN, 
            mixer: Mixer,
            gamma=0.99, 
            tau=0.01, 
            batch_size=64, 
            lr=0.0005, 
            optimizer: torch.optim.Optimizer = None,
            train_policy: Policy = None, 
            test_policy: Policy = None, 
            memory: EpisodeMemory = None, 
            double_qlearning=True,
            device: torch.device = None,
            logger: Logger=None,
            update_frequency=200,
            use_soft_update=True,
            train_interval=1,
            ir_module: ir.IRModule=None
        ):
        parameters = list(qnetwork.parameters()) + list(mixer.parameters())
        if optimizer is None:
            optimizer = torch.optim.Adam(parameters, lr=lr)
        super().__init__(
            qnetwork=qnetwork, 
            gamma=gamma, 
            tau=tau, 
            batch_size=batch_size, 
            optimizer=optimizer, 
            train_policy=train_policy, 
            test_policy=test_policy, 
            memory=memory,
            device=device,
            double_qlearning=double_qlearning,
            update_frequency=update_frequency,
            use_soft_update=use_soft_update,
            logger=logger,
            train_interval=train_interval
        )
        self._parameters = parameters
        self.mixer = mixer.to(self._device, non_blocking=True)
        self.target_mixer = deepcopy(mixer).to(self._device, non_blocking=True)
        if ir_module is not None:
            ir_module = ir_module.to(self._device)
        self._ir_module = ir_module
        

    def value(self, obs: Observation) -> float:
        qvalues = torch.max(self.compute_qvalues(obs), dim=-1).values
        qvalues = torch.unsqueeze(qvalues, dim=0)
        state = torch.from_numpy(obs.state).to(self._device, non_blocking=True)
        return self.mixer.forward(qvalues, state).item()

    def compute_loss(self, qvalues: torch.Tensor, qtargets: torch.Tensor, batch: EpisodeBatch) -> torch.Tensor:
        td_error = qtargets - qvalues
        mse = torch.mean(td_error ** 2)
        return mse

    def compute_targets(self, batch: TransitionsBatch) -> torch.Tensor:
        with torch.no_grad():
            target_next_qvalues = self._qtarget.forward(batch.obs_, batch.extras_)
            if self._double_qlearning:
                # Take the indices from the target network and the values from the current network
                # instead of taking both from the target network
                current_next_qvalues = self._qnetwork.forward(batch.obs_, batch.extras_)
                current_next_qvalues[batch.available_actions_ == 0.0] = -torch.inf
                indices = torch.argmax(current_next_qvalues, dim=-1, keepdim=True)
            else:
                target_next_qvalues[batch.available_actions_ == 0.0] = -torch.inf
                indices = torch.argmax(target_next_qvalues, dim=-1, keepdim=True)
        next_qvalues = torch.gather(target_next_qvalues, -1, indices).squeeze(-1)
        next_qvalues = self.target_mixer.forward(next_qvalues, batch.states_).squeeze()
        rewards = batch.rewards
        if self._ir_module is not None:
            ir = self._ir_module.intrinsic_reward(batch)
            self._train_logs["intrinsic_reward"] = ir.mean().item()
            rewards = rewards + ir
        targets = rewards + self.gamma * next_qvalues * (1 - batch.dones)
        return targets

    def _target_update(self, time_step: int):
        if time_step % self._update_frequency == 0:
            self._qtarget.load_state_dict(self._qnetwork.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _target_soft_update(self):
        for param, target_param in zip(self.mixer.parameters(), self.target_mixer.parameters()):
            new_value = (1-self._tau) * target_param.data + self._tau * param.data
            target_param.data.copy_(new_value, non_blocking=True)
        return super()._target_soft_update()
    
    def process_batch(self, batch: TransitionsBatch) -> TransitionsBatch:
        return batch

    def compute_qvalues(self, data: TransitionsBatch | Observation) -> torch.Tensor:
        match data:
            case TransitionsBatch() as batch:
                qvalues = self._qnetwork.forward(batch.obs, batch.extras)
                qvalues = torch.gather(qvalues, index=batch.actions, dim=-1)
                qvalues = qvalues.squeeze(dim=-1)
                return self.mixer.forward(qvalues, batch.states).squeeze()
            case Observation() as obs:
                obs_data = torch.from_numpy(obs.data).to(self._device, non_blocking=True).unsqueeze(0)
                obs_extras = torch.from_numpy(obs.extras).to(self._device, non_blocking=True).unsqueeze(0)
                qvalues = self._qnetwork.forward(obs_data, obs_extras)
                return qvalues.squeeze(dim=0)
            case _: raise ValueError(f"Invalid input data type {data.__class__.__name__} for 'compute_qvalues'")

    def to(self, device: torch.device):
        self.mixer.to(device)
        self.target_mixer.to(device)
        if self._ir_module is not None:
            self._ir_module.to(device)
        return super().to(device)
    
    def save(self, to_directory: str):
        super().save(to_directory)
        self.mixer.save(to_directory)
    
    def load(self, from_directory: str):
        super().load(from_directory)
        self.mixer.load(from_directory)


    def summary(self) -> dict[str, ]:
        return {
            **super().summary(),
            "mixer": self.mixer.summary(),
            "ir_module": self._ir_module.summary() if self._ir_module is not None else None
        }

    @classmethod
    def from_summary(cls, summary: dict[str, ]):
        from marl.qlearning import mixers
        device = defaults_to(summary.get("device"), get_device)
        summary["device"] = device
        summary["mixer"] = mixers.from_summary(summary['mixer']).to(device)
        ir_module = summary.get("ir_module")
        if ir_module is not None:
            summary["ir_module"] = ir.from_summary(summary["ir_module"]).to(device)
        return super().from_summary(summary)
    
    


    
class RecurrentMixedDQN(DQN):
    def __init__(
            self, 
            qnetwork: RecurrentNN, 
            mixer: Mixer,
            gamma=0.99, 
            tau=0.01, 
            batch_size=64, 
            lr=0.0005, 
            optimizer: torch.optim.Optimizer = None,
            train_policy: Policy = None, 
            test_policy: Policy = None, 
            memory: EpisodeMemory = None, 
            double_qlearning=True,
            device: torch.device = None,
            logger: Logger=None
        ):
        parameters = list(qnetwork.parameters()) + list(mixer.parameters())
        if optimizer is None:
            optimizer = torch.optim.RMSprop(parameters, lr=lr, alpha=0.99, eps=1e-5)
        super().__init__(
            qnetwork=qnetwork, 
            gamma=gamma, 
            tau=tau, 
            batch_size=batch_size, 
            optimizer=optimizer, 
            train_policy=train_policy, 
            test_policy=test_policy, 
            memory=defaults_to(memory, lambda: EpisodeMemory(5000)),
            device=device
        )
        self._parameters = parameters
        self.mixer = mixer.to(self._device, non_blocking=True)
        self.target_mixer = deepcopy(mixer).to(self._device, non_blocking=True)
        self._hidden_state = None
        self._saved_hidden_state = None
        self.logger = logger

        # Type hinting
        self._qnetwork: RecurrentNN = self._qnetwork
        self._qtarget: RecurrentNN = self._qtarget
        self._memory: EpisodeMemory = self._memory
        self._update_count = 0
        self._double_qlearning = double_qlearning

    def value(self, obs: Observation) -> float:
        hidden_state = self._hidden_state
        qvalues = torch.max(self.compute_qvalues(obs), dim=-1).values
        qvalues = torch.unsqueeze(qvalues, dim=0)
        self._hidden_state = hidden_state
        state = torch.from_numpy(obs.state).to(self._device, non_blocking=True)
        return self.mixer.forward(qvalues, state).item()

    def compute_loss(self, qvalues: torch.Tensor, qtargets: torch.Tensor, batch: EpisodeBatch) -> torch.Tensor:
        error = qtargets - qvalues
        masked_error = error * batch.masks
        criterion = masked_error ** 2
        loss = torch.sum(criterion) / torch.sum(batch.masks)
        return loss

    @torch.no_grad()
    def compute_targets(self, batch: EpisodeBatch) -> torch.Tensor:
        target_next_qvalues = self._qtarget.forward(batch.obs_, batch.extras_)[0]
        # Remove the first observation from the batch since we are taking the next observations
        target_next_qvalues = target_next_qvalues[1:]
        if self._double_qlearning:
            # Take the indices from the target network and the values from the current network
            # instead of taking both from the target network
            current_next_qvalues = self._qnetwork.forward(batch.obs_, batch.extras_)[0]
            current_next_qvalues = current_next_qvalues[1:]
            current_next_qvalues[batch.available_actions_ == 0.0] = -torch.inf
            indices = torch.argmax(current_next_qvalues, dim=-1, keepdim=True)
        else:
            target_next_qvalues[batch.available_actions_ == 0.0] = -torch.inf
            indices = torch.argmax(target_next_qvalues, dim=-1, keepdim=True)
        next_qvalues = torch.gather(target_next_qvalues, -1, indices).squeeze(-1)
        next_qvalues = self.target_mixer.forward(next_qvalues, batch.states_)
        targets = batch.rewards + self.gamma * next_qvalues * (1 - batch.dones)
        return targets

    def _target_soft_update(self):
        self._update_count += 1
        if self._update_count % 200 == 0:
            self._qtarget.load_state_dict(self._qnetwork.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        # for param, target_param in zip(self.mixer.parameters(), self.target_mixer.parameters()):
        #     new_value = (1-self._tau) * target_param.data + self._tau * param.data
        #     target_param.data.copy_(new_value, non_blocking=True)
        # return super()._target_soft_update()
    
    def process_batch(self, batch: EpisodeBatch) -> EpisodeBatch:
        return batch

    def compute_qvalues(self, data: EpisodeBatch | Observation) -> torch.Tensor:
        match data:
            case EpisodeBatch() as batch:
                qvalues = self._qnetwork.forward(batch.obs, batch.extras, None)[0]
                qvalues = torch.gather(qvalues, index=batch.actions, dim=-1)
                qvalues = qvalues.squeeze(dim=-1)
                return self.mixer.forward(qvalues, batch.states)
            case Observation() as obs:
                obs_data = torch.from_numpy(obs.data).to(self._device, non_blocking=True).unsqueeze(0)
                obs_extras = torch.from_numpy(obs.extras).to(self._device, non_blocking=True).unsqueeze(0)
                qvalues, self._hidden_state = self._qnetwork.forward(obs_data, obs_extras, self._hidden_state)
                return qvalues.squeeze(dim=0)
            case _: raise ValueError(f"Invalid input data type {data.__class__.__name__} for 'compute_qvalues'")

    
    def after_train_step(self, *_):
        self._train_policy.update()

    def after_train_episode(self, episode_num: int, episode: Episode):
        self._memory.add(episode)
        self.update()

    def before_tests(self, time_step: int):
        self._saved_hidden_state = self._hidden_state
        return super().before_tests(time_step)
    
    def after_tests(self, time_step: int, episodes):
        self._hidden_state = self._saved_hidden_state
        return super().after_tests(time_step, episodes)
        
    def before_train_episode(self, episode_num: int):
        self._hidden_state = None
        return super().before_train_episode(episode_num)

    def before_test_episode(self, time_step: int, test_num: int):
        self._hidden_state = None
        return super().before_test_episode(time_step, test_num)

    def to(self, device: torch.device):
        self.mixer.to(device)
        self.target_mixer.to(device)
        return super().to(device)
    
    def save(self, to_directory: str):
        super().save(to_directory)
        self.mixer.save(to_directory)
    
    def load(self, from_directory: str):
        super().load(from_directory)
        self.mixer.load(from_directory)


    def summary(self) -> dict[str, ]:
        return {
            **super().summary(),
            "mixer": self.mixer.summary()
        }

    @classmethod
    def from_summary(cls, summary: dict[str, ]):
        from marl.qlearning import mixers
        summary["mixer"] = mixers.from_summary(summary['mixer'])
        return super().from_summary(summary)
    
    