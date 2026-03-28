from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
import torch
from marlenv import Episode

from marl.models import Trainer

if TYPE_CHECKING:
    from marl.models import ActorCritic


@dataclass
class Reinforce(Trainer):
    """Vanilla policy gradient algorithm."""

    lr: float
    n_agents: int
    ac: "ActorCritic"
    gamma: float
    computation_method: Literal["monte_carlo", "td1"]

    def __init__(
        self,
        lr: float,
        n_agents: int,
        actor_critic: "ActorCritic",
        gamma: float,
        returns_computation: Literal["monte_carlo", "td1"] = "monte_carlo",
    ):
        super().__init__()
        self.lr = lr
        self.n_agents = n_agents
        self.ac = actor_critic
        self.gamma = gamma
        self._optim = torch.optim.AdamW(actor_critic.parameters(), lr=lr)
        self.computation_method = returns_computation

    def compute_td1_returns(self, episode: Episode):
        obs = torch.from_numpy(episode.next_obs).to(self.device)
        extras = torch.from_numpy(episode.next_extras).to(self.device)
        next_values = self.ac.value(obs, extras)
        return episode.rewards + self.gamma * next_values.numpy(force=True)

    def update_episode(self, episode: Episode, episode_num: int, time_step: int) -> dict[str, Any]:
        match self.computation_method:
            case "monte_carlo":
                G = torch.from_numpy(episode.compute_returns(self.gamma))
            case "td1":
                next_obs = torch.from_numpy(episode.next_obs).to(self.device)
                next_extras = torch.from_numpy(episode.next_extras).to(self.device)
                rewards = torch.from_numpy(episode.rewards).to(self.device)
                next_values = self.ac.value(next_obs, next_extras)
                G = rewards + self.gamma * next_values
            case other:
                raise ValueError(f"Invalid returns computation method: {other}")
        obs = torch.from_numpy(np.array(episode.obs)).to(self.device)
        extras = torch.from_numpy(np.array(episode.extras)).to(self.device)
        with torch.no_grad():
            values = self.ac.value(obs, extras)
            adv = G - values
        actions = torch.from_numpy(np.array(episode.actions)).to(self.device)
        log_probs = self.ac.log_probs(obs, extras, actions)
        loss = -(log_probs * adv.detach()).mean()
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()
        return {
            "loss": loss.item(),
            "returns_mean": G.mean().item(),
            "adv_mean": adv.mean().item(),
            "log_probs_mean": log_probs.mean().item(),
        }

    def make_agent(self):
        from marl.agents import SimpleActor

        return SimpleActor(self.ac)
