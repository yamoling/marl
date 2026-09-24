from dataclasses import KW_ONLY, dataclass
from typing import Any, Literal

import torch
from marlenv import Episode

from marl.models import Trainer
from marl.models.batch import EpisodeBatch
from marl.models.nn import Actor, Critic


@dataclass
class Reinforce(Trainer):
    """Vanilla policy gradient algorithm."""

    n_agents: int
    actor: Actor
    critic: Critic
    _: KW_ONLY
    lr: float = 1e-4
    returns_computation_method: Literal["monte_carlo", "td1"] = "monte_carlo"

    def __post_init__(self):
        """Set up optimization for both the policy and its learned baseline. @ai-edited"""
        super().__post_init__()
        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise ValueError("Reinforce currently requires non-recurrent actor and critic networks")
        self._optim = torch.optim.AdamW([*self.actor.parameters(), *self.critic.parameters()], lr=self.lr)

    def compute_returns(self, batch: EpisodeBatch) -> torch.Tensor:
        """Bootstrap truncated MC rollouts and mask terminal TD(1) targets. @ai-generated"""
        with torch.no_grad():
            next_values = self.critic.value(batch.next_obs.flatten(0, 1), batch.next_extras.flatten(0, 1)).unflatten(
                0, batch.next_obs.shape[:2]
            )
            match self.returns_computation_method:
                case "monte_carlo":
                    return batch.compute_mc_returns(self.gamma, next_values[-1])
                case "td1":
                    return batch.compute_td1_returns(self.gamma, next_values)
                case other:
                    raise ValueError(f"Invalid returns computation method: {other}")

    def update_episode(self, episode: Episode, episode_num: int, time_step: int) -> dict[str, Any]:
        """Train the masked policy and value baseline against episode returns. @ai-edited"""
        batch = EpisodeBatch([episode], device=self.device).for_individual_learners()
        returns = self.compute_returns(batch)
        obs = batch.obs.flatten(0, 1)
        extras = batch.extras.flatten(0, 1)
        values = self.critic.value(obs, extras).unflatten(0, batch.obs.shape[:2])
        advantages = returns - values.detach()
        policy = self.actor.policy(obs, extras, available_actions=batch.available_actions.flatten(0, 1))
        log_probs = policy.log_prob(batch.actions.flatten(0, 1)).unflatten(0, batch.obs.shape[:2])
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = torch.nn.functional.mse_loss(values, returns)
        loss = actor_loss + critic_loss
        self._optim.zero_grad()
        loss.backward()
        if self.grad_norm_clipping is not None:
            torch.nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], self.grad_norm_clipping)
        self._optim.step()
        return {
            "loss": loss.item(),
            "critic_loss": critic_loss.item(),
            "returns_mean": returns.mean().item(),
            "adv_mean": advantages.mean().item(),
            "log_probs_mean": log_probs.mean().item(),
        }

    def make_agent(self):
        """Use the same actor interface for behaviour and training. @ai-edited"""
        from marl.agents import SimpleAgent

        return SimpleAgent(self.actor)
