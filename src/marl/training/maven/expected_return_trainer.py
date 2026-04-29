from collections import deque
from dataclasses import KW_ONLY, dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
from marlenv import Episode, Observation
from marl.agents import DiscreteOneHotAgent
from marl.models import QNetwork, Trainer


@dataclass
class ExpectedReturnTrainer(Trainer[npt.NDArray[np.int64]]):
    nn: QNetwork
    _: KW_ONLY
    undiscounted: bool = True
    optimiser_type: Literal["adam", "rms"] = "adam"
    lr: float = 1e-5
    memory_size: int = 512
    batch_size: int = 64
    n_epochs: int = 8

    def __post_init__(self):
        super().__post_init__()
        match self.optimiser_type:
            case "adam":
                self._optimiser = torch.optim.Adam(self.nn.parameters(), lr=self.lr)
            case "rms":
                self._optimiser = torch.optim.RMSprop(self.nn.parameters(), lr=self.lr)
            case other:
                raise ValueError(f"optimiser_type should either be 'adam' or 'rms' but got {other}")
        self._memory = deque[tuple[Observation, np.ndarray, float]](maxlen=self.memory_size)
        self._loss = torch.nn.MSELoss()

    def update_episode(self, episode: Episode, episode_num: int, time_step: int) -> dict[str, float]:
        initial_obs = next(episode.transitions()).obs
        action = episode["maven-noise"][0]
        total_return = np.sum(episode.rewards).item()
        self._memory.append((initial_obs, action, total_return))
        if len(self._memory) < self.batch_size:
            return {}
        losses = []
        for _ in range(self.n_epochs):
            indices = np.random.randint(0, len(self._memory), size=self.batch_size)
            batch = [self._memory[i] for i in indices]
            obs = torch.tensor([x[0].data for x in batch], dtype=torch.float32, device=self.device)
            extras = torch.tensor([x[0].extras for x in batch], dtype=torch.float, device=self.device)
            actions = torch.tensor([x[1] for x in batch], dtype=torch.long, device=self.device)
            returns = torch.tensor([x[2] for x in batch], dtype=torch.float, device=self.device)

            predicted_returns = self.nn(obs, extras)
            predicted_returns = predicted_returns.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)
            loss = self._loss.forward(predicted_returns, returns)
            losses.append(loss.item())
            self._optimiser.zero_grad()
            loss.backward()
            self._optimiser.step()
        return {
            "mean_loss": np.mean(losses).item(),
            "min_loss": np.min(losses).item(),
            "max_loss": np.max(losses).item(),
        }

    def make_agent(self):
        return DiscreteOneHotAgent()
