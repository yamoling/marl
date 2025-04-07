from dataclasses import dataclass
from typing import Optional

import torch

from marl.models import IRModule, QNetwork
from marl.models.batch import Batch
from marlenv.utils import Schedule


@dataclass
class ToMIR(IRModule):
    """
    Theory of Mind Intrinsic Reward Module.

    The agents query the

    """

    def __init__(
        self,
        qnetwork: QNetwork,
        ir_weight: float | Schedule = 1e-2,
        *,
        agent_id_indices: Optional[list[int]] = None,
        n_agents: Optional[int] = None,
    ):
        super().__init__()
        self.qnetwork = qnetwork
        self.agent_id_indices = agent_id_indices
        if isinstance(ir_weight, (float, int)):
            ir_weight = Schedule.constant(ir_weight)
        self.ir_weight = ir_weight
        if agent_id_indices is None:
            assert n_agents is not None
            self.n_agents = n_agents
        else:
            assert n_agents is None
            self.n_agents = len(agent_id_indices)

    def compute(self, batch: Batch):
        with torch.no_grad():
            qvalues = self.qnetwork.forward(batch.obs, batch.extras)
            # Transpose from (batch, agents, ...) to (agent, batch, ...)
            extras = batch.extras.transpose(0, 1)
            obs = batch.obs.transpose(0, 1)
            qvalues = qvalues.transpose(0, 1)
            total_ir = torch.zeros(batch.size, dtype=torch.float32, device=batch.device)
            for agent, (agent_qvalues, agent_obs, agent_extras) in enumerate(zip(qvalues, obs, extras)):
                extras_perspective = self.from_perspective(agent_extras, agent)
                extras_perspective = extras_perspective.unsqueeze(1)
                extras_perspective = extras_perspective.repeat((1, self.n_agents, 1))

                obs_dims = (1 for _ in agent_obs.shape[1:])
                agent_obs = agent_obs.unsqueeze(1)
                agent_obs = agent_obs.repeat((1, self.n_agents, *obs_dims))
                others_qvalues = self.qnetwork.forward(batch.obs, extras_perspective)

                agent_actions = batch.actions[:, agent]
                agent_actions = agent_actions.unsqueeze(1)

                agent_qvalue = torch.gather(agent_qvalues, -1, agent_actions)  # The qvalue of the action taken by the current agent
                del agent_qvalues  # prevent misuse

                agent_actions = agent_actions.repeat(1, self.n_agents)
                agent_actions = agent_actions.unsqueeze(-1)

                others_qvalue = torch.gather(others_qvalues, -1, agent_actions).squeeze(-1)
                del others_qvalues  # prevent misuse

                diff = others_qvalue - agent_qvalue
                # With the max operator, it is guarantee that the minimal IR is 0
                # since the agent itself is alsao in the list of others, hence the diff is 0
                ir = torch.max(diff, dim=1).values
                total_ir += ir
            return total_ir * self.ir_weight

    def from_perspective(self, extras: torch.Tensor, agent_id: int):
        if self.agent_id_indices is None:
            return extras
        extras = extras.clone().detach()
        one_hot = torch.zeros(self.n_agents, dtype=torch.float32)
        one_hot[agent_id] = 1.0
        extras[:, self.agent_id_indices] = one_hot
        return extras
