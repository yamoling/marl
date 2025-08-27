import torch
from dataclasses import dataclass


@dataclass
class RunningMeanStd:
    """Credits to https://github.com/openai/random-network-distillation"""

    mean: torch.Tensor
    variance: torch.Tensor
    clip_min: float
    clip_max: float

    def __init__(
        self,
        shape: tuple[int, ...] = (1,),
        clip_min: float = -5,
        clip_max: float = 5,
    ):
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.variance = torch.ones(shape, dtype=torch.float32)
        self.count = 0
        self.clip_min = clip_min
        self.clip_max = clip_max

    @property
    def std(self):
        return self.variance**0.5

    def update(self, batch: torch.Tensor):
        batch_mean = torch.mean(batch, dim=0)
        # The unbiased=False is important to match the original numpy implementation that does not apply Bessel's correction
        # https://pytorch.org/docs/stable/generated/torch.var.html
        batch_var = torch.var(batch, dim=0, unbiased=False)
        batch_size = batch.shape[0]

        # Variance calculation from Wikipedia
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        tot_count = self.count + batch_size
        batch_ratio = batch_size / tot_count

        delta = batch_mean - self.mean
        m_a = self.variance * self.count
        m_b = batch_var * batch_size
        M2 = m_a + m_b + delta**2 * self.count * batch_ratio
        ## end of variance calculation from Wikipedia ##

        self.mean = self.mean + delta * batch_ratio
        self.variance = M2 / tot_count
        self.count = tot_count

    def normalise(self, batch: torch.Tensor, update=True) -> torch.Tensor:
        if update:
            self.update(batch)
        normalised = (batch - self.mean) / (self.variance + 1e-8) ** 0.5
        return torch.clip(normalised, min=self.clip_min, max=self.clip_max)

    def to(self, device: torch.device):
        self.mean = self.mean.to(device)
        self.variance = self.variance.to(device)
