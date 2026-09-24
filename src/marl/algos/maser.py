import math
from dataclasses import KW_ONLY, dataclass, field

import torch
import torch.nn.functional as F

from marl.models import NN, Batch, Mixer, RecurrentQNetwork
from marl.models.batch import EpisodeBatch, TransitionBatch
from marl.utils.tuning import tuning

from .dqn import DQN


@dataclass
class ActionableRepresentation(NN):
    """
    Representation transform φ of MASER, shared by all agents.

    It maps an agent observation (and its extras, which typically identify the agent) to an embedding in which
    Euclidean distances approximate the Q-function-based actionable distance (Section 4.3 of the paper).

    @ai-generated
    """

    obs_shape: tuple[int, ...]
    extras_size: int
    embedding_size: int
    _: KW_ONLY
    hidden_size: int = 128
    output_shape: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        self.output_shape = (self.embedding_size,)
        super().__post_init__()
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(math.prod(self.obs_shape) + self.extras_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.embedding_size),
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        """Embed observations of shape (*dims, *obs_shape) into (*dims, embedding_size). @ai-generated"""
        obs = obs.flatten(start_dim=obs.ndim - len(self.obs_shape))
        return self.nn.forward(torch.cat((obs, extras), dim=-1))

    def __hash__(self):
        return id(self)


def at_timestep(tensor: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    """
    Select, for every (episode, agent) pair, the item of `tensor` at the given time step.

    Args:
        tensor: shape (time, batch, n_agents, *rest).
        timesteps: shape (batch, n_agents).

    Returns:
        A tensor of shape (1, batch, n_agents, *rest) that broadcasts against `tensor`.

    @ai-generated
    """
    n_rest = tensor.ndim - timesteps.ndim - 1
    indices = timesteps.view(1, *timesteps.shape, *([1] * n_rest))
    return torch.take_along_dim(tensor, indices, dim=0)


@dataclass(unsafe_hash=True)
class MASER(DQN[Mixer]):
    """
    MASER: Multi-Agent Reinforcement Learning with Subgoals Generated from Experience Replay Buffer.

    Value-decomposition (QMIX in the paper) learner for sparse-reward tasks. For each sampled episode, every agent
    receives as subgoal the observation of that episode maximising `α max_u Q^i + (1 - α) Q_tot / N` (eq. 1).
    Agents then get an intrinsic reward equal to minus the distance to their subgoal in a learned actionable
    representation (eq. 2, 6, 7). The mixer is trained on the proxy reward `R_t` (eq. 3) and each agent's utility on
    its individual reward `r^i_t` (eq. 4). An entropy "episodic correction" loss is applied after each agent's
    subgoal time step (eq. 8). All losses are combined as in eq. 10.

    Differences with the paper:
        - φ is shared by all agents (like the utility networks) and also receives the observation extras.
        - The episodic correction only considers the available actions.
        - With transition replay, subgoals are selected from the sampled transitions rather than within each episode;
          episodic correction is omitted because the sampled transitions have no temporal ordering. Recurrent networks
          require episode replay.

    The paper uses `alpha=0.5`, `intrinsic_weight=0.03` and `1e-3` for the three auxiliary loss weights, with
    RMSProp (lr=5e-4), 32 episodes per batch, a 5000-episodes buffer and a hard target update every 200 episodes.

    Paper: https://proceedings.mlr.press/v162/jeon22a.html

    @ai-generated
    """

    _: KW_ONLY
    alpha: float = field(default=0.5, metadata=tuning(0.0, 1.0))
    """Weight of the individual Q-value relative to the total Q-value when selecting subgoals (eq. 1)."""
    intrinsic_weight: float = field(default=0.03, metadata=tuning(1e-3, 1.0, log=True))
    """λ, weight of the intrinsic rewards (eq. 3 and 4)."""
    individual_loss_weight: float = 1e-3
    """λ_I, weight of the individual TD losses."""
    correction_loss_weight: float = 1e-3
    """λ_E, weight of the episodic correction (entropy) losses."""
    representation_loss_weight: float = 1e-3
    """λ_D, weight of the representation losses."""
    representation_hidden_size: int = 128

    def __post_init__(self):
        """Validate the setting and add the representation network to the optimiser. @ai-generated"""
        if self.mixer is None:
            raise ValueError("MASER requires a mixer (QMIX in the paper)")
        if self.mixer.n_objectives != 1 or self.qnetwork.is_multi_objective:
            raise ValueError("MASER only supports scalar rewards")
        if self.memory.update_on_transitions and isinstance(self.qnetwork, RecurrentQNetwork):
            raise ValueError("MASER requires episode replay for recurrent Q-networks")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        super().__post_init__()
        self.representation = ActionableRepresentation(
            self.qnetwork.obs_shape,
            self.qnetwork.extras_size,
            self.qnetwork.n_actions,
            hidden_size=self.representation_hidden_size,
        )
        settings = self.optimiser.param_groups[0].copy()
        settings.pop("params")
        self.optimiser.add_param_group({"params": list(self.representation.parameters()), **settings})

    @property
    def name(self):
        return f"MASER-{self.qnetwork.name}"

    def _select_subgoals(self, greedy_qvalues: torch.Tensor, qtotal: torch.Tensor, masked_indices: torch.Tensor):
        """
        Eq. (1): time step of each agent's subgoal within each episode.

        Args:
            greedy_qvalues: `max_u Q^i(o^i_t, u)` with shape (time, batch, n_agents).
            qtotal: `Q_tot(o_t, u_t)` with shape (time, batch).
            masked_indices: padding mask with shape (time, batch).

        Returns:
            The subgoal time steps with shape (batch, n_agents).

        @ai-generated
        """
        n_agents = greedy_qvalues.shape[-1]
        scores = self.alpha * greedy_qvalues + (1 - self.alpha) * qtotal.unsqueeze(-1) / n_agents
        scores = scores.masked_fill(masked_indices.unsqueeze(-1), -torch.inf)
        return scores.argmax(dim=0)

    def _design_rewards(self, rewards: torch.Tensor, greedy_qvalues: torch.Tensor, intrinsic_rewards: torch.Tensor):
        """
        Eq. (3) and (4): proxy reward of the mixer and individual rewards of the agents.

        Args:
            rewards: extrinsic rewards with shape (time, batch).
            greedy_qvalues: `max_u Q^i(o^i_t, u)` with shape (time, batch, n_agents).
            intrinsic_rewards: `r^{i-int}_t` with shape (time, batch, n_agents).

        Returns:
            The proxy rewards (time, batch) and the individual rewards (time, batch, n_agents).

        @ai-generated
        """
        proxy_rewards = rewards + self.intrinsic_weight * intrinsic_rewards.mean(dim=-1)
        contributions = torch.softmax(greedy_qvalues, dim=-1)
        individual_rewards = contributions * proxy_rewards.unsqueeze(-1) + self.intrinsic_weight * intrinsic_rewards
        return proxy_rewards, individual_rewards

    def _compute_maser_targets(self, batch: Batch, proxy_rewards: torch.Tensor, individual_rewards: torch.Tensor):
        """
        Compute the targets of the mixed value (from the proxy reward) and of the individual utilities (from the
        individual rewards), sharing the same bootstrapped actions.

        @ai-generated
        """
        if isinstance(batch, TransitionBatch):
            next_obs, next_extras = batch.next_obs, batch.next_extras
            masks = None
        else:
            next_obs, next_extras = batch.all_obs, batch.all_extras
            masks = batch.all_masks
        next_qvalues = self.qtarget.batch_qvalues(next_obs, next_extras, masks=masks)
        if self.double_qlearning:
            self.qnetwork.eval()
            qvalues_for_index = self.qnetwork.batch_qvalues(next_obs, next_extras, masks=masks)
            self.qnetwork.train()
        else:
            qvalues_for_index = next_qvalues
        if isinstance(batch, EpisodeBatch):
            next_qvalues = next_qvalues[1:]
            qvalues_for_index = qvalues_for_index[1:]
        indices = qvalues_for_index.masked_fill(~batch.next_available_actions, -torch.inf).argmax(dim=-1, keepdim=True)
        next_values = next_qvalues.gather(-1, indices).squeeze(-1)
        next_total = self.target_mixer.forward_batch(next_values, batch, next_qvalues, indices.squeeze(-1), is_next=True)
        next_total = next_total.reshape(proxy_rewards.shape)
        terminal = batch.dones | batch.masked_indices
        total_targets = proxy_rewards + self.gamma * next_total.masked_fill(terminal, 0)
        individual_targets = individual_rewards + self.gamma * next_values.masked_fill(terminal.unsqueeze(-1), 0)
        return total_targets, individual_targets

    @staticmethod
    def _uniform_kl(qvalues: torch.Tensor, available_actions: torch.Tensor):
        """
        Eq. (8-9): KL divergence between softmax(Q^i) and the uniform distribution over the available actions.

        @ai-generated
        """
        # A finite fill value keeps both the forward and the backward pass free of NaNs (0 * -inf).
        logits = qvalues.masked_fill(~available_actions, -1e9)
        log_probs = F.log_softmax(logits, dim=-1)
        negative_entropy = (log_probs.exp() * log_probs).masked_fill(~available_actions, 0).sum(dim=-1)
        n_available = available_actions.sum(dim=-1).clamp_min(1)
        return negative_entropy + torch.log(n_available.float())

    def train(self, time_step: int, batch: Batch):
        """
        Select subgoals, shape the rewards, then minimise the MASER loss of eq. (10).

        @ai-generated
        """
        all_qvalues, qtotal = self._compute_qvalues(batch)
        qtotal = qtotal.reshape(batch.masks.shape)
        chosen_qvalues = all_qvalues.gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)
        agent_masks = batch.masks.unsqueeze(-1)
        n_agents = all_qvalues.shape[-2]
        episodic = isinstance(batch, EpisodeBatch)

        # Subgoal generation (Section 4.1) and actionable distance (eq. 6)
        with torch.no_grad():
            qvalues = all_qvalues.detach()
            greedy_actions = qvalues.masked_fill(~batch.available_actions, -torch.inf).argmax(dim=-1, keepdim=True)
            greedy_qvalues = qvalues.gather(-1, greedy_actions).squeeze(-1)
            if episodic:
                subgoal_timesteps = self._select_subgoals(greedy_qvalues, qtotal.detach(), batch.masked_indices)
                goal_qvalues = at_timestep(qvalues, subgoal_timesteps)
            else:
                scores = self.alpha * greedy_qvalues + (1 - self.alpha) * qtotal.detach().unsqueeze(-1) / n_agents
                subgoal_indices = scores.argmax(dim=0)  # one sampled transition per agent
                agent_indices = torch.arange(n_agents, device=qvalues.device)
                goal_qvalues = qvalues[subgoal_indices, agent_indices].unsqueeze(0)
            actionable_distances = 1 - F.cosine_similarity(qvalues, goal_qvalues, dim=-1)

        # Representation loss (eq. 7)
        embeddings = self.representation.forward(batch.obs, batch.extras)
        if episodic:
            goal_embeddings = at_timestep(embeddings, subgoal_timesteps)
        else:
            goal_embeddings = embeddings[subgoal_indices, agent_indices].unsqueeze(0)
        distances = torch.linalg.vector_norm(embeddings - goal_embeddings, dim=-1)
        representation_loss = ((distances - actionable_distances) ** 2 * agent_masks).sum() / batch.n_items

        # Reward design (eq. 2, 3 and 4)
        with torch.no_grad():
            intrinsic_rewards = -distances.detach() * agent_masks
            proxy_rewards, individual_rewards = self._design_rewards(batch.rewards, greedy_qvalues, intrinsic_rewards)
            total_targets, individual_targets = self._compute_maser_targets(batch, proxy_rewards, individual_rewards)

        # Total and individual TD losses
        td_loss, td_error = self._compute_td_loss(qtotal, total_targets, batch)
        individual_loss = (((chosen_qvalues - individual_targets) * agent_masks) ** 2).sum() / batch.n_items

        # Eq. (8) needs ordered episode histories; an unordered transition sample cannot identify later steps.
        if episodic:
            timesteps = torch.arange(all_qvalues.shape[0], device=all_qvalues.device).view(-1, 1, 1)
            after_subgoal = (timesteps >= subgoal_timesteps.unsqueeze(0)) & ~batch.masked_indices.unsqueeze(-1)
            kl = self._uniform_kl(all_qvalues, batch.available_actions)
            correction_loss = (kl * after_subgoal).sum() / batch.n_items
        else:
            correction_loss = qtotal.new_zeros(())

        loss = (
            td_loss
            + self.individual_loss_weight * individual_loss
            + self.correction_loss_weight * correction_loss
            + self.representation_loss_weight * representation_loss
        )
        self.optimiser.zero_grad()
        loss.backward()
        logs = {
            "td-loss": td_loss.item(),
            "individual-loss": individual_loss.item(),
            "correction-loss": correction_loss.item(),
            "representation-loss": representation_loss.item(),
            "loss": loss.item(),
            "intrinsic-reward": (intrinsic_rewards.sum() / (batch.n_items * n_agents)).item(),
        }
        if episodic:
            logs["subgoal-timestep"] = subgoal_timesteps.float().mean().item()
        else:
            logs["subgoal-index"] = subgoal_indices.float().mean().item()
        if self.grad_norm_clipping is not None:
            parameters = [*self.target_updater.parameters, *self.representation.parameters()]
            logs["grad_norm"] = torch.nn.utils.clip_grad_norm_(parameters, self.grad_norm_clipping).item()
        self.optimiser.step()
        return logs | self.memory.update(time_step, td_error=td_error)
