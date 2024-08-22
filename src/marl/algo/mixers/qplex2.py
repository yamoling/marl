from marlenv import RLEnv
import torch

from marl.models.nn import Mixer
from marl.nn.layers import AbsLayer


class QPlex2(Mixer):
    """Duplex dueling without attention mechanism (this differs from the original paper)."""

    def __init__(
        self,
        n_agents: int,
        n_actions: int,
        state_size: int,
        adv_hypernet_embed: int,
        transformation=True,
    ):
        super().__init__(n_agents)
        self.n_actions = n_actions
        self.state_size = state_size
        self.do_transformation = transformation

        self.w_extractor = torch.nn.Sequential(
            torch.nn.Linear(state_size, adv_hypernet_embed),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(adv_hypernet_embed, adv_hypernet_embed),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(adv_hypernet_embed, n_agents),
            AbsLayer(),
        )
        self.b_extractor = torch.nn.Sequential(
            torch.nn.Linear(state_size, adv_hypernet_embed),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(adv_hypernet_embed, adv_hypernet_embed),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(adv_hypernet_embed, n_agents),
        )

        self.lambda_extractor = torch.nn.Sequential(
            torch.nn.Linear(state_size + n_actions * n_agents, adv_hypernet_embed),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(adv_hypernet_embed, adv_hypernet_embed),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(adv_hypernet_embed, n_agents),
            AbsLayer(),
        )

    @classmethod
    def from_env(cls, env: RLEnv, adv_hypernet_embed: int = 64, transformation=True):
        assert len(env.state_shape) == 1
        return QPlex2(env.n_agents, env.n_actions, env.state_shape[0], adv_hypernet_embed, transformation)

    def transformation(
        self,
        states: torch.Tensor,
        qvalues: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transformation step as shown in Figure 1 of the paper."""
        # Positive weights (last layer is an AbsLayer)
        w = self.w_extractor(states) + 1e-10
        # Unconstrained bias
        b = self.b_extractor(states)
        qvalues = qvalues * w + b
        values = values * w + b
        advantages = qvalues - values
        return values, advantages

    def dueling_mixing(
        self,
        advantages: torch.Tensor,
        values: torch.Tensor,
        states: torch.Tensor,
        one_hot_actions: torch.Tensor,
    ) -> torch.Tensor:
        v_tot = torch.sum(values, dim=-1)

        lambdas_input = torch.concat([states, one_hot_actions], dim=-1)
        lambdas = self.lambda_extractor(lambdas_input)
        a_tot = lambdas * advantages
        a_tot = torch.sum(a_tot, dim=-1)
        return a_tot + v_tot

    def forward(
        self,
        qvalues: torch.Tensor,
        states: torch.Tensor,
        one_hot_actions: torch.Tensor,
        all_qvalues: torch.Tensor,
        *_args,
        **_kwargs,
    ) -> torch.Tensor:
        # Ignore n_actions, n_objective dimensions
        *dims, _, n_objectives = qvalues.shape
        qvalues = qvalues.view(-1, self.n_agents)
        states = states.reshape(-1, self.state_size)
        one_hot_actions = one_hot_actions.view(-1, self.n_actions * self.n_agents)
        # State value is the maximal qvalue
        values = all_qvalues.view(-1, self.n_agents, self.n_actions).max(dim=-1).values

        if self.do_transformation:
            values, advantages = self.transformation(states, qvalues, values)
        else:
            advantages = qvalues - values
        # I don't know why we need to detach the values here but they do it in the original code
        # and it seems not to work when we remove it.
        advantages = advantages.detach()
        q_tot = self.dueling_mixing(advantages, values, states, one_hot_actions)
        return q_tot.view(*dims, n_objectives)
