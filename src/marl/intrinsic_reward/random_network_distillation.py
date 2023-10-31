from dataclasses import dataclass
from typing import Optional
from serde import serde

import torch
import os

from marl.models import Batch
from marl.nn import randomize
from marl.nn.model_bank import CNN
from marl.utils.schedule import ConstantSchedule, Schedule
from marl.utils.stats import RunningMeanStd

from .ir_module import IRModule


@serde
@dataclass
class RandomNetworkDistillation(IRModule):
    obs_shape: tuple[int, ...]
    extras_shape: tuple[int, ...]
    features_size: int
    lr: float
    clip_value: float
    update_ratio: float
    running_mean_warmup: int
    ir_weight: Schedule

    def __init__(self, obs_shape: tuple[int, ...], extras_shape: tuple[int, ...], feature_size: int=512, lr:float=1e-4, clip_value: float=1.0, update_ratio: float=0.25, running_mean_warmup: int=64, ir_weight: Optional[Schedule]=None):
        super().__init__()
        self.obs_shape = obs_shape
        self.extras_shape = extras_shape
        self.features_size = feature_size
        self.lr = lr
        self.clip_value = clip_value
        self.update_ratio = update_ratio
        self.running_mean_warmup = running_mean_warmup
        if ir_weight is None:
            ir_weight = ConstantSchedule(1.0)
        self.ir_weight = ir_weight

        self.target = CNN(self.obs_shape, self.extras_shape, output_shape=(self.features_size,))
        # Add an extra layer to the predictor to make it more difficult to predict the target
        self.predictor_head = CNN(self.obs_shape, self.extras_shape, output_shape=(self.features_size,))
        self.predictor_tail = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(512, self.features_size))
        self.optimizer = torch.optim.Adam(list(self.predictor_head.parameters()) + list(self.predictor_tail.parameters()), lr=self.lr)

        # Initialize the running mean and std (section 2.4 of the article)
        self._running_reward = RunningMeanStd()
        self._running_obs = RunningMeanStd(shape=self.obs_shape)
        self._update_count = 0
        self._warmup_duration = self.running_mean_warmup



    def compute(self, batch: Batch) -> torch.Tensor:
        self._update_count += 1

        # Compute the embedding and the squared error
        with torch.no_grad():
            target_features = self.target.forward(batch.obs_, batch.extras_)
        predicted_features = self.predictor_head.forward(batch.obs_, batch.extras_)
        predicted_features = self.predictor_tail.forward(predicted_features)
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

    def randomize(self):
        self.target.randomize()
        self.predictor_head.randomize()
        randomize(torch.nn.init.xavier_uniform_, self.predictor_tail)

    def save(self, to_directory: str):
        os.makedirs(to_directory, exist_ok=True)
        target_path = os.path.join(to_directory, "target.weights")
        torch.save(self.target.state_dict(), target_path)
        predictor_head_path = os.path.join(to_directory, "predictor_head.weights")
        torch.save(self.predictor_head.state_dict(), predictor_head_path)
        predictor_tail_path = os.path.join(to_directory, "predictor_tail.weights")
        torch.save(self.predictor_tail.state_dict(), predictor_tail_path)

    def load(self, from_directory: str):
        target_path = os.path.join(from_directory, "target.weights")
        self.target.load_state_dict(torch.load(target_path))
        predictor_head_path = os.path.join(from_directory, "predictor_head.weights")
        self.predictor_head.load_state_dict(torch.load(predictor_head_path))
        predictor_tail_path = os.path.join(from_directory, "predictor_tail.weights")
        self.predictor_tail.load_state_dict(torch.load(predictor_tail_path))
        return self