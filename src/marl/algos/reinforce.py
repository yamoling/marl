from dataclasses import KW_ONLY, dataclass
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

    n_agents: int
    ac: "ActorCritic"
    _: KW_ONLY
    lr: float = 1e-4
    returns_computation_method: Literal["monte_carlo", "td1"] = "monte_carlo"

    def __post_init__(self):
        super().__post_init__()
        self._optim = torch.optim.AdamW(self.ac.parameters(), lr=self.lr)

    def compute_td1_returns(self, episode: Episode):
        obs = torch.from_numpy(episode.next_obs).to(self.device)
        extras = torch.from_numpy(episode.next_extras).to(self.device)
        next_values = self.ac.value(obs, extras)
        return episode.rewards + self.gamma * next_values.numpy(force=True)

    def update_episode(self, episode: Episode, episode_num: int, time_step: int) -> dict[str, Any]:
        match self.returns_computation_method:
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
        from marl.agents import SimpleAgent

        return SimpleAgent(self.ac)
