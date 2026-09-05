from collections import deque
from dataclasses import KW_ONLY, dataclass, field

import torch
import torch.nn.functional as F
from marlenv import Episode

from marl.models import NN, Batch, EpisodeMemory
from marl.nn import mixers

from .dqn import DQN


@dataclass(unsafe_hash=True)
class ExternalStateTransitionModel(NN):
    """Predict the next external state from the current state and joint action.

    @ai-generated
    """

    state_size: int
    state_extras_size: int
    n_agents: int
    n_actions: int
    _: KW_ONLY
    hidden_size: int = 128
    output_shape: tuple[int, ...] = field(init=False)
    external_state_size: int

    def __post_init__(self):
        """Build the recurrent external-state transition model.

        @ai-generated
        """
        self.output_shape = (self.external_state_size,)
        super().__post_init__()
        input_size = self.state_size + self.state_extras_size + self.n_agents * self.n_actions
        self.input_layer = torch.nn.Linear(input_size, self.hidden_size)
        self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size)
        self.output_layer = torch.nn.Linear(self.hidden_size, self.external_state_size)

    def forward_with_history(self, states: torch.Tensor, states_extras: torch.Tensor, actions: torch.Tensor):
        """Predict factual outcomes and return the hidden state preceding each action.

        @ai-generated
        """
        inputs = torch.cat((states, states_extras, actions.flatten(start_dim=-2)), dim=-1)
        is_transition_batch = inputs.dim() == 2
        if is_transition_batch:
            inputs = inputs.unsqueeze(0)
        hidden, _ = self.gru(F.relu(self.input_layer(inputs)))
        previous_hidden = torch.cat((torch.zeros_like(hidden[:1]), hidden[:-1]), dim=0)
        predictions = self.output_layer(hidden)
        if is_transition_batch:
            return predictions.squeeze(0), previous_hidden.squeeze(0)
        return predictions, previous_hidden

    def counterfactual(
        self,
        states: torch.Tensor,
        states_extras: torch.Tensor,
        actions: torch.Tensor,
        previous_hidden: torch.Tensor,
    ):
        """Predict one-step outcomes for candidate joint actions using factual history.

        @ai-generated
        """
        n_candidates = actions.shape[-3]
        states = states.unsqueeze(-2).expand(*states.shape[:-1], n_candidates, states.shape[-1])
        states_extras = states_extras.unsqueeze(-2).expand(
            *states_extras.shape[:-1], n_candidates, states_extras.shape[-1]
        )
        inputs = torch.cat((states, states_extras, actions.flatten(start_dim=-2)), dim=-1)
        hidden = previous_hidden.unsqueeze(-2).expand(*previous_hidden.shape[:-1], n_candidates, self.hidden_size)
        outputs, _ = self.gru(
            F.relu(self.input_layer(inputs)).reshape(1, -1, self.hidden_size),
            hidden.reshape(1, -1, self.hidden_size),
        )
        return self.output_layer(outputs.squeeze(0)).reshape(*actions.shape[:-2], self.external_state_size)


@dataclass(unsafe_hash=True)
class LAIES(DQN):
    """QMIX with Lazy Agents Avoidance through Influencing External States.

    `external_state_indices` identifies the flattened global-state features that
    agents should be encouraged to influence.

    Paper: https://proceedings.mlr.press/v202/liu23ac.html

    @ai-generated
    """

    _: KW_ONLY
    external_state_indices: tuple[int, ...]
    beta_idi: float = 1.0
    beta_cdi: float = 1.0
    cdi_samples: int = 4
    estm_hidden_size: int = 128
    estm_lr: float = 3e-4
    estm_updates: int = 1
    intrinsic_reward_clip: float | None = 1.0
    intrinsic_anneal_steps: int = 0
    intrinsic_anneal_window: int = 32

    def __post_init__(self):
        """Initialize QMIX and its external-state transition model.

        @ai-generated
        """
        if self.mixer.n_objectives != 1:
            raise ValueError("LAIES currently supports only scalar rewards")
        if not self.external_state_indices:
            raise ValueError("LAIES requires at least one external-state feature index")
        if min(self.external_state_indices) < 0 or max(self.external_state_indices) >= self.mixer.state_size:
            raise ValueError(f"External-state indices must be in [0, {self.mixer.state_size})")
        if self.cdi_samples < 1 or self.estm_updates < 1:
            raise ValueError("cdi_samples and estm_updates must be positive")
        if self.memory_size == "auto":
            self.memory_size = 5000
        super().__post_init__()
        self.memory = EpisodeMemory(self.memory_size)
        self.estm = ExternalStateTransitionModel(
            self.mixer.state_size,
            self.mixer.state_extras_size,
            self.qnetwork.n_agents,
            self.qnetwork.n_actions,
            external_state_size=len(self.external_state_indices),
            hidden_size=self.estm_hidden_size,
        )
        self.estm_optimiser = torch.optim.Adam(self.estm.parameters(), lr=self.estm_lr)
        self._recent_returns = deque[float](maxlen=self.intrinsic_anneal_window)
        self._anneal_start = None
        self._anneal_scale = 1.0

    @property
    def name(self):
        return f"LAIES-{self.qnetwork.name}"

    def _external_states(self, states: torch.Tensor):
        return states[..., self.external_state_indices]

    def update_episode(self, episode: Episode, episode_num: int, time_step: int):
        """Track the extrinsic return in order to anneal the intrinsic rewards, then run the DQN update.

        The paper anneals the intrinsic rewards once the mean extrinsic return becomes positive, i.e. once
        the team wins more often than it loses. `intrinsic_anneal_steps` is the duration of that linear
        decay; annealing is disabled when it is left at 0.

        @ai-generated
        """
        if self.intrinsic_anneal_steps > 0:
            self._recent_returns.append(sum(episode.score))
            is_full = len(self._recent_returns) == self._recent_returns.maxlen
            mean_return = sum(self._recent_returns) / max(len(self._recent_returns), 1)
            if self._anneal_start is None and is_full and mean_return > 0:
                self._anneal_start = time_step
            if self._anneal_start is not None:
                progress = (time_step - self._anneal_start) / self.intrinsic_anneal_steps
                self._anneal_scale = max(0.0, 1.0 - progress)
        return super().update_episode(episode, episode_num, time_step)

    def _train_estm(self, batch: Batch):
        """Fit the ESTM to observed external-state transitions.

        @ai-generated
        """
        if batch.states.shape[-1] != self.mixer.state_size:
            raise ValueError("LAIES requires flattened global states")
        target = self._external_states(batch.next_states)
        loss_value = 0.0
        for _ in range(self.estm_updates):
            prediction, _ = self.estm.forward_with_history(
                batch.states, batch.states_extras, batch.one_hot_actions.float()
            )
            item_loss = F.mse_loss(prediction, target, reduction="none").mean(dim=-1)
            loss = (item_loss * batch.masks).sum() / batch.masks.sum().clamp_min(1)
            self.estm_optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.estm.parameters(), 1.0)
            self.estm_optimiser.step()
            loss_value = loss.item()
        return loss_value

    def _intrinsic_rewards(self, batch: Batch):
        """Compute the paper's individual and collaborative diligence rewards.

        @ai-generated
        """
        actions = batch.one_hot_actions.float()
        available = batch.available_actions
        with torch.no_grad():
            factual, previous_hidden = self.estm.forward_with_history(batch.states, batch.states_extras, actions)
            eye = torch.eye(batch.n_actions, device=self.device)
            individual = torch.zeros_like(batch.rewards)
            for agent in range(batch.n_agents):
                candidates = (
                    actions.unsqueeze(-3).expand(*actions.shape[:-2], batch.n_actions, *actions.shape[-2:]).clone()
                )
                candidates[..., agent, :] = eye
                predictions = self.estm.counterfactual(batch.states, batch.states_extras, candidates, previous_hidden)
                valid = available[..., agent, :] & ~actions[..., agent, :].bool()
                count = valid.sum(dim=-1, keepdim=True)
                counterfactual = (predictions * valid.unsqueeze(-1)).sum(dim=-2) / count.clamp_min(1)
                diligence = F.mse_loss(factual, counterfactual, reduction="none").mean(dim=-1)
                individual += diligence * (count.squeeze(-1) > 0)

            sampled = torch.multinomial(
                available.float().reshape(-1, batch.n_actions), self.cdi_samples, replacement=True
            )
            sampled = sampled.reshape(*available.shape[:-2], batch.n_agents, self.cdi_samples).movedim(-1, -2)
            factual_actions = batch.actions.unsqueeze(-2)
            has_counterfactual = available.sum(dim=-1).prod(dim=-1) > 1
            same_as_factual = (sampled == factual_actions).all(dim=-1) & has_counterfactual.unsqueeze(-1)
            while same_as_factual.any():
                replacements = torch.multinomial(
                    available.float().reshape(-1, batch.n_actions), self.cdi_samples, replacement=True
                )
                replacements = replacements.reshape(*available.shape[:-2], batch.n_agents, self.cdi_samples).movedim(
                    -1, -2
                )
                sampled = torch.where(same_as_factual.unsqueeze(-1), replacements, sampled)
                same_as_factual = (sampled == factual_actions).all(dim=-1) & has_counterfactual.unsqueeze(-1)
            joint_candidates = F.one_hot(sampled, batch.n_actions).float()
            joint_predictions = self.estm.counterfactual(
                batch.states, batch.states_extras, joint_candidates, previous_hidden
            )
            collaborative = F.mse_loss(factual, joint_predictions.mean(dim=-2), reduction="none").mean(dim=-1)

            intrinsic = self.beta_idi * individual + self.beta_cdi * collaborative
            if self.intrinsic_reward_clip is not None:
                intrinsic = intrinsic.clamp(max=self.intrinsic_reward_clip)
            intrinsic = intrinsic * self._anneal_scale
        return intrinsic, individual, collaborative

    def train(self, time_step: int, batch: Batch):
        """Train ESTM, add diligence rewards, then perform the QMIX update.

        @ai-generated
        """
        estm_loss = self._train_estm(batch)
        intrinsic, individual, collaborative = self._intrinsic_rewards(batch)
        mask_sum = batch.masks.sum().clamp_min(1)
        batch.rewards = batch.rewards + intrinsic
        logs = super().train(time_step, batch)
        return logs | {
            "estm-loss": estm_loss,
            "idi": ((individual * batch.masks).sum() / mask_sum).item(),
            "cdi": ((collaborative * batch.masks).sum() / mask_sum).item(),
            "intrinsic-reward": ((intrinsic * batch.masks).sum() / mask_sum).item(),
            "intrinsic-scale": self._anneal_scale,
        }
