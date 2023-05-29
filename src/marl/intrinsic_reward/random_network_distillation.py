import torch
from marl.utils import get_device, defaults_to
from marl.models import Batch
from marl.nn.model_bank import CNN
from marl.utils.stats import RunningMeanStd
from marl.utils.schedule import Schedule, ConstantSchedule
from marl.utils.summarizable import Summarizable
from .ir_module import IRModule

class RandomNetworkDistillation(IRModule):
    def __init__(
            self, 
            obs_shape: tuple[int, ...], 
            extras_shape: tuple[int, ...], 
            features_size=512, 
            lr=1e-4, 
            clip_value=1, 
            update_ratio=0.25,
            running_mean_warmup=64,
            ir_weight: Schedule = None,
            device: torch.device = None
        ):
        super().__init__()

        self._features_size = features_size
        self._obs_shape = obs_shape
        self._extras_shape = extras_shape
        self._lr = lr
        self._clip_max = clip_value
        self._update_ratio = update_ratio
        self._target = CNN(obs_shape, extras_shape, output_shape=(features_size, ))
        self._target.randomize("orthogonal")
        # Add an extra layer to the predictor to make it more difficult to predict the target
        self._predictor_head = CNN(obs_shape, extras_shape, output_shape=(features_size, ))
        self._predictor_tail = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(512, features_size)
        )
        parameters = list(self._predictor_head.parameters()) + list(self._predictor_tail.parameters())
        self._optimizer = torch.optim.Adam(parameters, lr=lr)
        self._device = defaults_to(device, get_device)
        if ir_weight is None:
            ir_weight = ConstantSchedule(1.)
        self._ir_weight = ir_weight

        self._target.randomize("orthogonal")
        self._predictor_head.randomize("orthogonal")
        self._target.to(self._device)
        self._predictor_head.to(self._device)
        self._predictor_tail.to(self._device)

        # Initialize the running mean and std (section 2.4 of the article)
        self._running_reward = RunningMeanStd().to(self._device)
        self._running_obs = RunningMeanStd(shape=obs_shape).to(self._device)
        self._update_count = 0
        self._warmup_duration = running_mean_warmup


    def intrinsic_reward(self, batch: Batch) -> torch.Tensor:
        self._update_count += 1

        # Compute the embedding and the squared error
        with torch.no_grad():
            target_features = self._target.forward(batch.obs_, batch.extras_)
        predicted_features = self._predictor_head.forward(batch.obs_, batch.extras_)
        predicted_features = self._predictor_tail.forward(predicted_features)
        squared_error = torch.pow(target_features - predicted_features,  2)
        # Reshape the error such that it is a vector of shape (batch_size, -1) to be able to sum over batch size even if there are multiple agents
        squared_error = squared_error.view(batch.size, -1)
        with torch.no_grad():
            intrinsic_reward = torch.sum(squared_error, dim=-1)
            # self._running_reward.update(intrinsic_reward)
            # intrinsic_reward = self._running_reward.normalize(intrinsic_reward)
        
        # Randomly mask some of the features and perform the optimization
        masks = torch.rand_like(squared_error) < self._update_ratio
        loss = torch.sum(squared_error * masks) / torch.sum(masks)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        
        if self._update_count < self._warmup_duration:
            return torch.zeros(batch.size, dtype=torch.float32).to(self._device)
        
        intrinsic_reward = torch.clip(intrinsic_reward, max=self._clip_max)
        intrinsic_reward = self._ir_weight.value * intrinsic_reward
        self._ir_weight.update()
        return intrinsic_reward
    

    def to(self, device: torch.device):
        self._target = self._target.to(device, non_blocking=True)
        self._predictor_head = self._predictor_head.to(device, non_blocking=True)
        self._predictor_tail = self._predictor_tail.to(device, non_blocking=True)
        self._running_obs = self._running_obs.to(device)
        self._running_reward = self._running_reward.to(device)
        self._device = device
        return self

    def update(self):
        pass
        

    def summary(self) -> dict[str, ]:
        return {
            **super().summary(),
            "obs_shape": self._obs_shape,
            "extras_shape": self._extras_shape,
            "features_size": self._features_size,
            "lr": self._lr,
            "clip_value": self._clip_max,
            "update_ratio": self._update_ratio,
            "running_mean_warmup": self._warmup_duration,
            "ir_weight": self._ir_weight.summary()
        }
    

    @classmethod
    def from_summary(cls, summary: dict[str, ]) -> Summarizable:
        from marl.utils import schedule
        ir_weight = summary.get("ir_weight")
        if ir_weight is not None:
            summary["ir_weight"] = schedule.from_summary(ir_weight)
        return super().from_summary(summary)