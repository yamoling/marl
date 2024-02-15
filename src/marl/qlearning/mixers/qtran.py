import torch
from marl.models.nn import Mixer


class QTRANBaseNet(torch.nn.Module):
    def __init__(self, state_size: int, n_agents: int, n_hidden_units: int):
        super().__init__(n_agents)
        self.nr_input_features = state_size + n_agents
        self.nr_hidden_units = n_hidden_units
        self.nr_agents = n_agents
        self.fc_net = torch.nn.Sequential(
            torch.nn.Linear(self.nr_input_features, self.nr_hidden_units),
            torch.nn.ELU(),
            torch.nn.Linear(self.nr_hidden_units, self.nr_hidden_units),
            torch.nn.ELU(),
        )
        self.joint_action_head = torch.nn.Linear(self.nr_hidden_units, 1)
        self.state_value_head = torch.nn.Linear(self.nr_hidden_units, 1)

    def forward(self, global_states, Q_values):
        batch_size = global_states.size(0)
        global_states = global_states.view(batch_size, -1)
        Q_values = Q_values.view(batch_size, self.nr_agents)
        x = torch.cat([global_states, Q_values], dim=-1)
        x = self.fc_net(x)
        return self.joint_action_head(x), self.state_value_head(x)


class QTRAN(Mixer):
    def __init__(self, n_agents: int, state_size: int):
        super(QTRAN, self).__init__(n_agents)
        self.central_value_network = QTRANBaseNet(
            state_size,
            n_agents,
            64,
        )

        raise NotImplementedError("QTRAN is not implemented yet")

    def optimizer_update(
        self,
        Q_local_values_real,
        Q_local_values_max,
        states,
        returns,
        subteam_indices,
    ):
        batch_size = Q_local_values_real.size(0)
        dummy = torch.zeros(3 * batch_size)
        real_detached_indices = dummy != dummy
        real_undetached_indices = dummy != dummy
        max_undetached_indices = dummy != dummy
        for counter in range(batch_size):
            real_detached_indices[counter] = True
            real_undetached_indices[counter + batch_size] = True
            max_undetached_indices[counter + 2 * batch_size] = True
        concat_states = torch.cat([states, states, states], dim=0)
        concat_Q_values = torch.cat(
            [Q_local_values_real.detach(), Q_local_values_real, Q_local_values_max],
            dim=0,
        )
        Q_total, V_total = self.global_value(concat_Q_values, concat_states)
        Q = Q_total[real_detached_indices].squeeze()
        V = V_total[real_undetached_indices].squeeze()
        Q_max = Q_total[max_undetached_indices].detach().squeeze()
        V_max = V_total[max_undetached_indices].squeeze()
        Q_transformed_real = Q_local_values_real.sum(1)
        Q_transformed_max = Q_local_values_max.sum(1)
        returns = returns.sum(1)
        loss_value = (Q - returns) ** 2
        loss_constraint1 = Q_transformed_max - Q_max + V_max**2
        constraint2_term = (Q_transformed_real - Q.detach() + V).unsqueeze(1)
        concat = torch.cat([constraint2_term, torch.zeros_like(constraint2_term)], dim=-1)
        loss_constraint2 = (concat.min(1)[0]) ** 2
        loss = (loss_value + loss_constraint1 + loss_constraint2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, self.clip_norm)
        self.optimizer.step()
        return (0, 0)

    def update(
        self,
        states,
        joint_actions,
        observations,
        returns,
        dones,
        old_probs,
        subteam_indices,
    ):
        batch_size = returns.size(0)
        new_batch_size = batch_size * self.nr_agents
        observations = observations.view(new_batch_size, -1)
        actions = joint_actions.view(new_batch_size, 1)
        Q_local_values, _ = self.Q_net(observations)
        Q_local_values_real = Q_local_values.gather(1, actions).squeeze()
        Q_local_values_max = Q_local_values.max(1)[0].squeeze()
        Q_local_values_real = Q_local_values_real.view(batch_size, self.nr_agents)
        Q_local_values_max = Q_local_values_max.view(batch_size, self.nr_agents)
        return self.optimizer_update(Q_local_values_real, Q_local_values_max, states, returns, subteam_indices)

    def global_value(self, Q_values, states):
        Q_values = Q_values.view(-1, 1, self.nr_subteams)
        return self.central_value_network(states, Q_values)

    def global_value_of(self, Q_values, states):
        return self.global_value(Q_values, states)[0]
