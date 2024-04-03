import torch

from typing import Literal, Any, Optional
from rlenv import Transition, Episode
from marl.models import Trainer, Mixer, EpisodeMemory, Policy, PrioritizedMemory
from marl.qlearning import MAICAlgo
from marl.models.batch import EpisodeBatch
from .qtarget_updater import TargetParametersUpdater, SoftUpdate
from rlenv.models import Observation
from marl.utils import defaults_to

from copy import deepcopy


class MAICTrainer(Trainer):
    def __init__(
        self,
        args,
        maic_algo: MAICAlgo,
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
        self.maic_algo = maic_algo
        self.target_algo = deepcopy(maic_algo)

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
        self.target_updater.add_parameters(maic_algo.parameters(), self.target_algo.parameters())
        if mixer is not None and self.target_mixer is not None:
            self.target_updater.add_parameters(mixer.parameters(), self.target_mixer.parameters())
        match optimiser:
            case "adam":
                self.optimiser = torch.optim.Adam(self.target_updater.parameters, lr=lr)
            case "rmsprop":
                self.optimiser = torch.optim.RMSprop(self.target_updater.parameters, lr=lr)
            case other:
                raise ValueError(f"Unknown optimiser: {other}. Expected 'adam' or 'rmsprop'.")

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

    def optimise_network(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        rewards = batch.rewards.squeeze(-1)[:-1, :]
        actions = batch.actions[:-1, :]
        terminated = batch.dones.squeeze(-1)[:-1, :].float()
        mask = batch.masks.squeeze(-1)[:-1, :].float()
        mask[1:, :] = mask[1:, :] * (1 - terminated[:-1, :])
        avail_actions = batch.available_actions 

        logs = []
        losses = []

        # Calculate estimated Q-Values
        mac_out = []
        self.maic_algo.init_hidden(self.batch_size)

        for t in range(batch._max_episode_len):
            agent_outs, returns_ = self.maic_algo.forward(batch.obs[t], batch.extras[t], test_mode=False)
            # qvalues = torch.gather(agent_outs, dim=-1, index=batch.actions).squeeze(-1)
            # if self.mixer is not None:
            #     qvalues = self.mixer.forward(qvalues, batch.states, batch.one_hot_actions, agent_outs)
            mac_out.append(agent_outs)
            if "logs" in returns_:
                logs.append(returns_["logs"])
                del returns_["logs"]
            losses.append(returns_)

        mac_out = torch.stack(mac_out, dim=0)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = torch.gather(mac_out[:-1, :], dim=3, index=actions).squeeze(3)  # Remove the last dim
        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_algo.init_hidden(self.batch_size)
        for t in range(batch._max_episode_len):
            target_agent_outs, _ = self.target_algo.forward(batch.obs[t], batch.extras[t], test_mode=False)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = torch.stack(target_mac_out[1:], dim=0)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[1:, :] == 0] = -torch.inf

        # Max over target Q-Values
        if self.double_qlearning:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -torch.inf
            cur_max_actions = mac_out_detach[1:, :].max(dim=3, keepdim=True)[1]
            target_max_qvals = torch.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None and self.target_mixer is not None:
            chosen_action_qvals = self.mixer.forward(chosen_action_qvals.unsqueeze(-1), batch.states_[:-1, :], batch.one_hot_actions[:-1, :], mac_out[:-1, :]).squeeze(-1)
            target_max_qvals = self.target_mixer.forward(target_max_qvals.unsqueeze(-1), batch.states_[1:, :], batch.one_hot_actions[1:, :], target_mac_out[1:, :]).squeeze(-1)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.gamma * (1 - terminated) * target_max_qvals
        
        # Td-error
        td_error = chosen_action_qvals - targets.detach()
        
        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error**2).sum() / mask.sum()

        external_loss, loss_dict = self._process_loss(losses, batch)
        loss += external_loss

        # Optimise
        logs = {"loss": float(loss.item())}
        self.optimiser.zero_grad()
        loss.backward()
        if self.grad_norm_clipping is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.maic_algo.parameters(), self.grad_norm_clipping)
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

    def to(self, device: torch.device):
        if self.mixer is not None:
            self.mixer.to(device)
        if self.target_mixer is not None:
            self.target_mixer.to(device)
        self.maic_algo.to(device)
        self.target_algo.to(device)
        self.device = device
        return self

    def randomize(self):
        self.maic_algo.randomize()
        self.target_algo.randomize()
        if self.mixer is not None:
            self.mixer.randomize()
        if self.target_mixer is not None:
            self.target_mixer.randomize()