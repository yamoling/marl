from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch

from marl.models import Batch
from marl.nn.model_bank import CNN
from marl.utils import get_device
from marl.utils.schedule import ConstantSchedule, Schedule
from marl.utils.stats import RunningMeanStd

from .ir_module import IRModule


@dataclass
class RandomNetworkDistillation(IRModule):
    obs_shape: tuple[int, ...]
    extras_shape: tuple[int, ...]
    features_size: int = 512
    lr: float = 1e-4
    clip_value: float = 1
    update_ratio: float = 0.25
    running_mean_warmup: int = 64
    ir_weight: Optional[Schedule] = None
    device_str: Literal["auto", "cpu", "cuda"] = "auto"

    def __post_init__(self):
        self.device = get_device(self.device_str)
        self.target = CNN(self.obs_shape, self.extras_shape, output_shape=(self.features_size,)).to(self.device)
        self.target.randomize("orthogonal")
        # Add an extra layer to the predictor to make it more difficult to predict the target
        cnn = CNN(self.obs_shape, self.extras_shape, output_shape=(self.features_size,))
        cnn.randomize("orthogonal")
        self.predictor = torch.nn.Sequential(cnn, torch.nn.ReLU(), torch.nn.Linear(512, self.features_size)).to(self.device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=self.lr)
        if self.ir_weight is None:
            self.ir_weight = ConstantSchedule(1.0)

        # Initialize the running mean and std (section 2.4 of the article)
        self._running_reward = RunningMeanStd().to(self.device)
        self._running_obs = RunningMeanStd(shape=self.obs_shape).to(self.device)
        self._update_count = 0
        self._warmup_duration = self.running_mean_warmup

    def compute(self, batch: Batch) -> torch.Tensor:
        self._update_count += 1

        # Compute the embedding and the squared error
        with torch.no_grad():
            target_features = self.target.forward(batch.obs_, batch.extras_)
        predicted_features = self.predictor.forward(batch.obs_, batch.extras_)
        squared_error = torch.pow(target_features - predicted_features, 2)
        # Reshape the error such that it is a vector of shape (batch_size, -1)
        # to be able to sum over batch size even if there are multiple agents
        squared_error = squared_error.view(batch.size, -1)
        with torch.no_grad():
            intrinsic_reward = torch.sum(squared_error, dim=-1)
            # self._running_reward.update(intrinsic_reward)
            # intrinsic_reward = self._running_reward.normalize(intrinsic_reward)

        # Randomly mask some of the features and perform the optimization
        masks = torch.rand_like(squared_error) < self.update_ratio
        loss = torch.sum(squared_error * masks) / torch.sum(masks)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self._update_count < self._warmup_duration:
            return torch.zeros(batch.size, dtype=torch.float32).to(self.device)

        intrinsic_reward = torch.clip(intrinsic_reward, max=self.clip_value)
        intrinsic_reward = self.ir_weight * intrinsic_reward
        self.ir_weight.update()
        return intrinsic_reward

    def to(self, device: torch.device):
        self.target = self.target.to(device, non_blocking=True)
        self.predictor_head = self.predictor_head.to(device, non_blocking=True)
        self.predictor_tail = self.predictor_tail.to(device, non_blocking=True)
        self._running_obs = self._running_obs.to(device)
        self._running_reward = self._running_reward.to(device)
        self.device = device
        return self

    def update(self):
        pass

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        from marl.utils import schedule

        data["ir_weight"] = schedule.from_dict(data["ir_weight"])
        return super().from_dict(data)
