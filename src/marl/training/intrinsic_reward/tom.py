import torch
from marl.models.batch import Batch
from dataclasses import dataclass
from marl.models import QNetwork, IRModule


@dataclass
class TOMIR(IRModule):
    def __init__(self, qnetwork: QNetwork, agent_id_indices: list[int]):
        super().__init__()
        self.qnetwork = qnetwork
        self.agent_id_indices = agent_id_indices
        self.n_agents = len(agent_id_indices)

    def compute(self, batch: Batch):
        with torch.no_grad():
            qvalues = self.qnetwork.forward(batch.obs, batch.extras)
            # Transpose from (batch, agents, ...) to (agent, batch, ...)
            extras = batch.extras.transpose(0, 1)
            obs = batch.obs.transpose(0, 1)
            qvalues = qvalues.transpose(0, 1)
            for agent, (agent_qvalues, agent_obs, agent_extras) in enumerate(zip(qvalues, obs, extras)):
                extras_perspective = self.from_perspective(agent_extras, agent)
                extras_perspective = extras_perspective.unsqueeze(1)
                extras_perspective = extras_perspective.repeat((1, self.n_agents, 1))
                agent_obs = agent_obs.unsqueeze(1).repeat((1, self.n_agents, 1))
                others_qvalues = self.qnetwork.forward(batch.obs, extras_perspective)
                diff = others_qvalues - agent_qvalues
                print(diff)
            perspectives = []
            for agent_num in range(self.n_agents):
                extras = self.from_perspective(batch.extras, agent_num)
            return 0.0

    def from_perspective(self, extras: torch.Tensor, agent_id: int):
        extras = extras.clone().detach()
        one_hot = torch.zeros(self.n_agents, dtype=torch.float32)
        one_hot[agent_id] = 1.0
        extras[:, self.agent_id_indices] = one_hot
        return extras
