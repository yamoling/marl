from abc import abstractmethod, ABC
import torch



class Mixer(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, qvalues: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        pass

class VDN(Mixer):
    def forward(self, qvalues: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        return torch.sum(qvalues, dim=-1)