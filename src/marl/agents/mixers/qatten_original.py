import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from marl.models import Mixer


class QattenOriginal(Mixer):
    def __init__(
        self,
        n_agents: int,
        n_actions: int,
        state_size: int,
        agent_state_size: int,
        mixer_embedding_dim: int = 32,
        hypernetwork_embed_size: int = 64,
        n_heads: int = 4,
        weighted_head: bool = False,
        nonlinear: bool = False,
    ):
        super().__init__(n_agents)

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_dim = state_size
        self.action_dim = n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim + 1
        self.unit_dim = agent_state_size

        self.attention_weight = Qatten_Weight(
            n_agents=n_agents,
            state_size=state_size,
            unit_dim=agent_state_size,
            n_actions=n_actions,
            n_heads=n_heads,
            mixing_embed_dim=mixer_embedding_dim,
            hypernet_embed=hypernetwork_embed_size,
            attend_reg_coef=0.001,
            weighted_head=weighted_head,
        )
        self.si_weight = DMAQ_SI_Weight(
            n_agents=n_agents,
            state_size=state_size,
            n_actions=n_actions,
            num_kernel=4,
            adv_hypernet_embed=64,
            nonlinear=nonlinear,
            adv_hypernet_layers=2,
        )

        self.V = nn.Sequential(
            nn.Linear(self.state_dim, mixer_embedding_dim),
            nn.ReLU(),
            nn.Linear(mixer_embedding_dim, 1),
        )

    def calc_v(self, agent_qs):
        agent_qs = agent_qs.view(-1, self.n_agents)
        v_tot = th.sum(agent_qs, dim=-1)
        return v_tot

    def calc_adv(self, agent_qs, states, actions, max_q_i):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_q_i = max_q_i.view(-1, self.n_agents)

        adv_q = (agent_qs - max_q_i).view(-1, self.n_agents).detach()

        adv_w_final = self.si_weight(states, actions)
        adv_w_final = adv_w_final.view(-1, self.n_agents)

        if self.args.is_minus_one:
            adv_tot = th.sum(adv_q * (adv_w_final - 1.0), dim=1)
        else:
            adv_tot = th.sum(adv_q * adv_w_final, dim=1)
        return adv_tot

    def calc(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        if is_v:
            v_tot = self.calc_v(agent_qs)
            return v_tot
        else:
            adv_tot = self.calc_adv(agent_qs, states, actions, max_q_i)
            return adv_tot

    def forward(self, agent_qs: th.Tensor, states, actions=None, max_q_i=None, is_v=False):
        bs = agent_qs.size(0)

        w_final, attend_mag_regs, head_entropies = self.attention_weight.forward(agent_qs, states)
        w_final = w_final.view(-1, self.n_agents) + 1e-10

        # V(s) instead of a bias for the last layers (c(s) in the paper, top right corner of the mixer in Figure 1.)
        v = self.V(states)
        v = v.view(-1, 1).repeat(1, self.n_agents)
        v /= self.n_agents

        agent_qs = agent_qs.view(-1, self.n_agents)
        agent_qs = w_final * agent_qs + v

        if not is_v:
            if max_q_i is None:
                raise ValueError()
            max_q_i = max_q_i.view(-1, self.n_agents)
            max_q_i = w_final * max_q_i + v

        y = self.calc(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=is_v)
        v_tot = y.view(bs, -1, 1)

        return v_tot, attend_mag_regs, head_entropies


class Qatten_Weight(nn.Module):
    def __init__(
        self,
        n_agents: int,
        state_size: int,
        unit_dim: int,
        n_actions: int,
        n_heads: int,
        mixing_embed_dim: int,
        hypernet_embed: int,
        attend_reg_coef: float,
        weighted_head: bool,
    ):
        super(Qatten_Weight, self).__init__()

        self.name = "qatten_weight"
        self.n_agents = n_agents
        self.state_dim = state_size
        self.unit_dim = unit_dim
        self.n_actions = n_actions
        self.sa_dim = self.state_dim + self.n_agents * self.n_actions
        self.n_head = n_heads  # attention head num

        self.embed_dim = mixing_embed_dim
        self.attend_reg_coef = attend_reg_coef
        self.weighted_head = weighted_head

        self.key_extractors = nn.ModuleList()
        self.query_extractors = nn.ModuleList()

        # self.attention = nn.MultiheadAttention(self.embed_dim, self.n_head)

        for i in range(self.n_head):  # multi-head attention
            selector_nn = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim, bias=False),
            )
            self.query_extractors.append(selector_nn)  # query
            if self.args.nonlinear:  # add qs
                self.key_extractors.append(nn.Linear(self.unit_dim + 1, self.embed_dim, bias=False))  # key
            else:
                self.key_extractors.append(nn.Linear(self.unit_dim, self.embed_dim, bias=False))  # key
        if weighted_head:
            self.hyper_w_head = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.n_head),
            )

    def forward(self, agent_qs, states):
        states = states.reshape(-1, self.state_dim)
        unit_states = states[:, : self.unit_dim * self.n_agents]  # get agent own features from state
        unit_states = unit_states.reshape(-1, self.n_agents, self.unit_dim)
        unit_states = unit_states.permute(1, 0, 2)

        agent_qs = agent_qs.view(-1, 1, self.n_agents)  # agent_qs: (batch_size, 1, agent_num)

        if self.args.nonlinear:
            unit_states = th.cat((unit_states, agent_qs.permute(2, 0, 1)), dim=2)
        # states: (batch_size, state_dim)
        all_head_selectors = [sel_ext(states) for sel_ext in self.query_extractors]
        # all_head_selectors: (head_num, batch_size, embed_dim)
        # unit_states: (agent_num, batch_size, unit_dim)
        all_head_keys = [[k_ext(enc) for enc in unit_states] for k_ext in self.key_extractors]
        # all_head_keys: (head_num, agent_num, batch_size, embed_dim)

        # calculate attention per head
        head_attend_logits = []
        head_attend_weights = []
        for curr_head_keys, curr_head_selector in zip(all_head_keys, all_head_selectors):
            # curr_head_keys: (agent_num, batch_size, embed_dim)
            # curr_head_selector: (batch_size, embed_dim)

            # (batch_size, 1, embed_dim) * (batch_size, embed_dim, agent_num)
            attend_logits = th.matmul(curr_head_selector.view(-1, 1, self.embed_dim), th.stack(curr_head_keys).permute(1, 2, 0))
            # attend_logits: (batch_size, 1, agent_num)
            # scale dot-products by size of key (from Attention is All You Need)
            scaled_attend_logits = attend_logits / np.sqrt(self.embed_dim)
            attend_weights = F.softmax(scaled_attend_logits, dim=2)  # (batch_size, 1, agent_num)

            head_attend_logits.append(attend_logits)
            head_attend_weights.append(attend_weights)

        head_attend = th.stack(head_attend_weights, dim=1)  # (batch_size, self.n_head, self.n_agents)
        head_attend = head_attend.view(-1, self.n_head, self.n_agents)

        # head_qs: [head_num, bs, 1]
        if self.args.weighted_head:
            w_head = th.abs(self.hyper_w_head(states))  # w_head: (bs, head_num)
            w_head = w_head.view(-1, self.n_head, 1).repeat(1, 1, self.n_agents)  # w_head: (bs, head_num, self.n_agents)
            head_attend *= w_head

        head_attend = th.sum(head_attend, dim=1)

        # regularize magnitude of attention logits
        attend_mag_regs = self.attend_reg_coef * sum((logit**2).mean() for logit in head_attend_logits)
        head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1).mean()) for probs in head_attend_weights]
        return head_attend, attend_mag_regs, head_entropies

    def forward_custom(self, agent_qs, states, actions):
        states = states.reshape(-1, self.state_dim)
        unit_states = states[:, : self.unit_dim * self.n_agents]  # get agent own features from state
        unit_states = unit_states.reshape(-1, self.n_agents, self.unit_dim)
        unit_states = unit_states.permute(1, 0, 2)

        agent_qs = agent_qs.view(-1, 1, self.n_agents)  # agent_qs: (batch_size, 1, agent_num)

        if self.args.nonlinear:
            unit_states = th.cat((unit_states, agent_qs.permute(2, 0, 1)), dim=2)

        all_queries = th.stack([sel_ext(states) for sel_ext in self.query_extractors])
        all_queries = all_queries.view(self.n_head, -1, 1, self.embed_dim)
        all_keys = th.stack([k_ext(unit_states) for k_ext in self.key_extractors])
        all_keys = all_keys.permute(0, 2, 3, 1)
        logits = th.matmul(all_queries, all_keys)
        scaled_logits = logits / self.embed_dim**0.5
        scaled_logits = th.permute(scaled_logits, (1, 0, 2, 3))
        head_attend = F.softmax(scaled_logits, dim=-1)
        head_attend = head_attend.view(-1, self.n_head, self.n_agents)

        # head_qs: [head_num, bs, 1]
        if self.args.weighted_head:
            w_head = th.abs(self.hyper_w_head(states))  # w_head: (bs, head_num)
            w_head = w_head.view(-1, self.n_head, 1).repeat(1, 1, self.n_agents)  # w_head: (bs, head_num, self.n_agents)
            head_attend *= w_head

        # regularize magnitude of attention logits
        attend_mag_regs = self.attend_reg_coef * sum(th.mean(z) for z in logits**2)
        weight_per_head = head_attend.permute(1, 0, 2)
        head_entropies = th.sum(-th.log(weight_per_head + 1e-8) * weight_per_head, dim=-1).mean(-1)
        head_attend = th.sum(head_attend, dim=1)
        return head_attend, attend_mag_regs, head_entropies


class DMAQ_SI_Weight(nn.Module):
    def __init__(
        self,
        n_agents: int,
        state_size: int,
        n_actions: int,
        num_kernel: int,
        adv_hypernet_embed: int,
        nonlinear: bool,
        adv_hypernet_layers: int,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_dim = state_size
        self.action_dim = n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim

        self.num_kernel = num_kernel

        self.key_extractors = nn.ModuleList()
        self.agents_extractors = nn.ModuleList()
        self.action_extractors = nn.ModuleList()

        for i in range(self.num_kernel):  # multi-head attention
            if adv_hypernet_layers == 1:
                self.key_extractors.append(nn.Linear(self.state_dim, 1))  # key
                self.agents_extractors.append(nn.Linear(self.state_dim, self.n_agents))  # agent
                self.action_extractors.append(nn.Linear(self.state_action_dim, self.n_agents))  # action
            elif adv_hypernet_layers == 2:
                self.key_extractors.append(
                    nn.Sequential(nn.Linear(self.state_dim, adv_hypernet_embed), nn.ReLU(), nn.Linear(adv_hypernet_embed, 1))
                )  # key
                self.agents_extractors.append(
                    nn.Sequential(nn.Linear(self.state_dim, adv_hypernet_embed), nn.ReLU(), nn.Linear(adv_hypernet_embed, self.n_agents))
                )  # agent
                self.action_extractors.append(
                    nn.Sequential(
                        nn.Linear(self.state_action_dim, adv_hypernet_embed), nn.ReLU(), nn.Linear(adv_hypernet_embed, self.n_agents)
                    )
                )  # action
            elif adv_hypernet_layers == 3:
                self.key_extractors.append(
                    nn.Sequential(
                        nn.Linear(self.state_dim, adv_hypernet_embed),
                        nn.ReLU(),
                        nn.Linear(adv_hypernet_embed, adv_hypernet_embed),
                        nn.ReLU(),
                        nn.Linear(adv_hypernet_embed, 1),
                    )
                )  # key
                self.agents_extractors.append(
                    nn.Sequential(
                        nn.Linear(self.state_dim, adv_hypernet_embed),
                        nn.ReLU(),
                        nn.Linear(adv_hypernet_embed, adv_hypernet_embed),
                        nn.ReLU(),
                        nn.Linear(adv_hypernet_embed, self.n_agents),
                    )
                )  # agent
                self.action_extractors.append(
                    nn.Sequential(
                        nn.Linear(self.state_action_dim, adv_hypernet_embed),
                        nn.ReLU(),
                        nn.Linear(adv_hypernet_embed, adv_hypernet_embed),
                        nn.ReLU(),
                        nn.Linear(adv_hypernet_embed, self.n_agents),
                    )
                )  # action
            else:
                raise Exception("Error setting number of adv hypernet layers.")

    def forward(self, states, actions):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        data = th.cat([states, actions], dim=1)

        all_head_key = [k_ext(states) for k_ext in self.key_extractors]
        all_head_agents = [k_ext(states) for k_ext in self.agents_extractors]
        all_head_action = [sel_ext(data) for sel_ext in self.action_extractors]

        head_attend_weights = []
        for curr_head_key, curr_head_agents, curr_head_action in zip(all_head_key, all_head_agents, all_head_action):
            x_key = th.abs(curr_head_key).repeat(1, self.n_agents) + 1e-10
            x_agents = F.sigmoid(curr_head_agents)
            x_action = F.sigmoid(curr_head_action)
            weights = x_key * x_agents * x_action
            head_attend_weights.append(weights)

        head_attend = th.stack(head_attend_weights, dim=1)
        head_attend = head_attend.view(-1, self.num_kernel, self.n_agents)
        head_attend = th.sum(head_attend, dim=1)

        return head_attend
