import torch

from typing import Literal, Any, Optional
from rlenv import Transition, Episode
from marl.models import Trainer, Mixer, EpisodeMemory, Policy, MAICNN
from marl.models.batch import EpisodeBatch
from .qtarget_updater import TargetParametersUpdater, SoftUpdate
from rlenv.models import Observation
from marl.utils import defaults_to

from copy import deepcopy


class MAICTrainer(Trainer):
    def __init__(
        self,
        args,
        maic_network: MAICNN,
        train_policy: Policy,
        memory: EpisodeMemory,
        gamma: float = 0.99,
        batch_size: int = 16,
        lr: float = 1e-4,
        optimiser: Literal["adam", "rmsprop"] = "adam",
        target_updater: Optional[TargetParametersUpdater] = None,
        double_qlearning: bool = False,
        mixer: Optional[Mixer] = None,
        train_interval: tuple[int, Literal["step", "episode"]] = (1, "episode"),
        grad_norm_clipping: Optional[float] = None,
    ):
        super().__init__(train_interval[1], train_interval[0])
        self.n_agents = args.n_agents
        self.maic_network = maic_network
        self.target_network = deepcopy(maic_network)

        self.policy = train_policy
        self.memory = memory
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_updater = defaults_to(target_updater, lambda: SoftUpdate(1e-2))
        self.double_qlearning = double_qlearning
        self.mixer = mixer
        self.target_mixer = deepcopy(mixer)

        self.update_num = 0

        # Parameters and optimiser
        self.grad_norm_clipping = grad_norm_clipping
        self.target_updater.add_parameters(maic_network.parameters(), self.target_network.parameters())
        if mixer is not None and self.target_mixer is not None:
            self.target_updater.add_parameters(mixer.parameters(), self.target_mixer.parameters())
        match optimiser:
            case "adam":
                self.optimiser = torch.optim.Adam(self.target_updater.parameters, lr=lr)
            case "rmsprop":
                self.optimiser = torch.optim.RMSprop(self.target_updater.parameters, lr=lr)
            case other:
                raise ValueError(f"Unknown optimiser: {other}. Expected 'adam' or 'rmsprop'.")

    def _update(self, episode_num: int, time_step: int):
        self.update_num += 1
        if self.update_num % self.update_interval != 0 or not self._can_update():
            return {}
        logs, td_error = self.optimise_network()
        logs = logs | self.policy.update(time_step)
        logs = logs | self.target_updater.update(episode_num)
        return logs
    
    def _can_update(self):
        return self.memory.can_sample(self.batch_size)

    def _next_state_value(self, batch: EpisodeBatch):
        # We use the all_obs_ and all_extras_ to handle the case of recurrent qnetworks that require the first element of the sequence.
        next_qvalues, _, _ = self.target_network.batch_forward(batch.all_obs_, batch.all_extras_)
        next_qvalues = next_qvalues[1:]
        # For double q-learning, we use the qnetwork to select the best action. Otherwise, we use the target qnetwork.
        if self.double_qlearning:
            qvalues_for_index, _, _ = self.maic_network.batch_forward(batch.all_obs_, batch.all_extras_)
            qvalues_for_index = qvalues_for_index[1:]
        else:
            qvalues_for_index = next_qvalues
        # Sum over the objectives
        qvalues_for_index = torch.sum(qvalues_for_index, -1)
        qvalues_for_index[batch.available_actions_ == 0.0] = -torch.inf
        indices = torch.argmax(qvalues_for_index, dim=-1, keepdim=True)
        indices = indices.unsqueeze(-1).repeat(*(1 for _ in indices.shape), batch.reward_size)
        next_values = torch.gather(next_qvalues, -2, indices).squeeze(-2)
        mixed_next_values = self.target_mixer.forward(next_values, batch.states_, batch.one_hot_actions, next_qvalues)
        return mixed_next_values

    def optimise_network(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        batch.multi_objective()

        # Calculate estimated Q-Values
        qvalues, logs, losses = self.maic_network.batch_forward(batch.obs, batch.extras)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = torch.gather(qvalues, dim=-2, index=batch.actions).squeeze(-2)
        mixed_qvalues = self.mixer.forward(chosen_action_qvals, batch.states, batch.one_hot_actions, qvalues)

        # Drop variables to prevent using them mistakenly
        del qvalues
        del chosen_action_qvals

        # Calculate the Q-Values necessary for the target
        next_values = self._next_state_value(batch)
   
        assert batch.rewards.shape == next_values.shape == batch.dones.shape == mixed_qvalues.shape == batch.masks.shape
        # Calculate 1-step Q-Learning targets
        qtargets = batch.rewards + self.gamma * (1 - batch.dones) * next_values
        
        # Td-error
        td_error = mixed_qvalues - qtargets.detach()

        # 0-out the targets that came from padded data
        td_error = td_error * batch.masks
        squared_error = td_error**2

        # Normal L2 loss, take mean over actual data
        loss = (squared_error).sum() / batch.masks.sum()

        external_loss, loss_dict = self._process_loss(losses, batch)
        loss += external_loss

        # Optimise
        logs = {"loss": float(loss.item())}
        self.optimiser.zero_grad()
        loss.backward()
        if self.grad_norm_clipping is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.maic_network.parameters(), self.grad_norm_clipping)
            logs["grad_norm"] = grad_norm.item()
        self.optimiser.step()

        return logs, td_error

    def _process_loss(self, losses: list, batch: EpisodeBatch):
        total_loss = 0
        loss_dict = {}
        for item in losses:
            for k, v in item.items():
                if str(k).endswith("loss"):
                    loss_dict[k] = loss_dict.get(k, 0) + v
                    total_loss += v
        for k in loss_dict.keys():
            loss_dict[k] /= batch._max_episode_len
        total_loss /= batch._max_episode_len
        return total_loss, loss_dict
    
    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        """
        Update to call after each step. Should be run when update_after_each == "step".

        Returns:
            dict[str, Any]: A dictionary of training metrics to log.
        """
        return {}

    def update_episode(self, episode: Episode, episode_num: int, time_step: int) -> dict[str, Any]:
        """
        Update to call after each episode. Should be run when update_after_each == "episode".

        Returns:
            dict[str, Any]: A dictionary of training metrics to log.
        """
        if not self.memory.update_on_transitions:
            self.memory.add(episode)
        return self._update(episode_num, time_step)

    def to(self, device: torch.device):
        if self.mixer is not None:
            self.mixer.to(device)
        if self.target_mixer is not None:
            self.target_mixer.to(device)
        self.maic_network.to(device)
        self.target_network.to(device)
        self.device = device
        return self

    def randomize(self):
        self.maic_network.randomize()
        self.target_network.randomize()
        if self.mixer is not None:
            self.mixer.randomize()
        if self.target_mixer is not None:
            self.target_mixer.randomize()