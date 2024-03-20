import os
import pickle
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from marl.utils import DotDic
from rlenv.models import Observation
from rlenv import Episode
from marl.models.batch import EpisodeBatch

from typing import Optional

from marl.models import RLAlgo, Policy, DRU
from marl.nn.model_bank import CNet

class CNetAlgo(RLAlgo):
    def __init__(self, opt, model: CNet, target: CNet, train_policy: Policy, test_policy: Policy):
        super().__init__()
        self.opt = opt
        self.model = model
        self.model_target = target

        self.train_policy = train_policy
        self.test_policy = test_policy
        self.policy = self.train_policy

        for p in self.model_target.parameters():
            p.requires_grad = False
        
        self.episodes_seen = 0
        self.dru = DRU(opt.game_comm_sigma, opt.model_comm_narrow, opt.game_comm_hard)

        self.optimizer = optim.RMSprop(
            params=model.get_params(), lr=opt.learningrate, momentum=opt.momentum)
        
        # TODO : Add Container with the history of select_action_and_comm + addition value not present in EpisodeBatch (to determine)
        self.training_mode = False
        
    
    ########## Save and Load content which is not in a Episode Batch ##########
    def create_episode(self):
        opt = self.opt

        episode = DotDic({})
        episode.steps = 0
        episode.ended = 0
        episode.r = torch.zeros(opt.game_nagents).float().to(self.device)
        episode.step_records = []

        return episode

    def create_step_record(self):
        opt = self.opt
        record = DotDic({})
        record.s_t = None
        record.r_t = torch.zeros(opt.game_nagents).to(self.device)
        record.terminal = 0

        record.agent_inputs = []

        # Track actions at time t per agent
        record.a_t = torch.zeros(opt.game_nagents, dtype=torch.long)
        if not opt.model_dial:
            record.a_comm_t = torch.zeros(opt.game_nagents, dtype=torch.long)

        # Track messages sent at time t per agent
        if opt.comm_enabled:
            comm_dtype = opt.model_dial and torch.float or torch.long
            comm_dtype = torch.float
            record.comm = torch.zeros(opt.game_nagents, opt.game_comm_bits, dtype=comm_dtype)
            if opt.model_dial and opt.model_target:
                record.comm_target = record.comm.clone()

        # Track hidden state per time t per agent
        record.hidden = torch.zeros(opt.game_nagents, opt.model_rnn_layers, opt.model_rnn_size)
        record.hidden_target = torch.zeros(opt.game_nagents, opt.model_rnn_layers, opt.model_rnn_size)

        # Track Q(a_t) and Q(a_max_t) per agent
        record.q_a_t = torch.zeros(opt.game_nagents)
        record.q_a_max_t = torch.zeros(opt.game_nagents)

        # Track Q(m_t) and Q(m_max_t) per agent
        if not opt.model_dial:
            record.q_comm_t = torch.zeros(opt.game_nagents)
            record.q_comm_max_t = torch.zeros(opt.game_nagents)

        return record
    ###########################################################################
        
    def reset(self):
        self.model.reset_parameters()
        self.model_target.reset_parameters()
        self.episodes_seen = 0
    
    def select_action_and_comm(self, obs: Observation, q, agent_idx: int, target=False):
        # eps-Greedy action selector
        opt = self.opt
        action = None
        action_value = None
        comm_action = None
        comm_vector = torch.zeros(opt.game_comm_bits)
        comm_value = None
        if not opt.model_dial:
            comm_value = torch.zeros(opt.bs)

        # Get action + comm
        action = self.policy.get_action(q[:opt.game_action_space], obs.available_actions[agent_idx])
        action_value = q[action]

        if not opt.model_dial:
            comm = self.policy.get_action(q[opt.game_action_space:opt.game_action_space_total], [1 for _ in range(opt.game_comm_bits)])
            action_value = q[comm + opt.game_action_space]
            comm_vector[comm] = 1
        else:
            comm_vector = self.dru.forward(q[opt.game_action_space:opt.game_action_space_total], self.training_mode)

        return (action, action_value), (comm_vector, comm_action, comm_value)

    def episode_loss(self, episode):
        opt = self.opt
        total_loss = torch.zeros(opt.bs)
        for b in range(opt.bs):
            b_steps = episode.steps[b].item()
            for step in range(b_steps):
                record = episode.step_records[step]
                for i in range(opt.game_nagents):
                    td_action = 0
                    td_comm = 0
                    r_t = record.r_t[b][i]
                    q_a_t = record.q_a_t[b][i]
                    q_comm_t = 0

                    if record.a_t[b][i].item() > 0:
                        if record.terminal[b].item() > 0:
                            td_action = r_t - q_a_t
                        else:
                            next_record = episode.step_records[step + 1]
                            q_next_max = next_record.q_a_max_t[b][i]
                            if not opt.model_dial and opt.model_avg_q:
                                q_next_max = (q_next_max + next_record.q_comm_max_t[b][i])/2.0
                            td_action = r_t + opt.gamma * q_next_max - q_a_t

                    if not opt.model_dial and record.a_comm_t[b][i].item() > 0:
                        q_comm_t = record.q_comm_t[b][i]
                        if record.terminal[b].item() > 0:
                            td_comm = r_t - q_comm_t
                        else:
                            next_record = episode.step_records[step + 1]
                            q_next_max = next_record.q_comm_max_t[b][i]
                            if opt.model_avg_q: 
                                q_next_max = (q_next_max + next_record.q_a_max_t[b][i])/2.0
                            td_comm = r_t + opt.gamma * q_next_max - q_comm_t

                    if not opt.model_dial:
                        loss_t = (td_action ** 2 + td_comm ** 2)
                    else:
                        loss_t = (td_action ** 2)
                    total_loss[b] = total_loss[b] + loss_t
        loss = total_loss.sum()
        loss = loss/(self.opt.bs * self.opt.game_nagents)
        return loss


    def learn_from_episode(self, episode):
        self.optimizer.zero_grad()
        loss = self.episode_loss(episode)
        loss.backward(retain_graph=not self.opt.model_know_share)
        clip_grad_norm_(parameters=self.model.get_params(), max_norm=10)
        self.optimizer.step()

        self.episodes_seen = self.episodes_seen + 1
        if self.episodes_seen % self.opt.step_target == 0:
            self.model_target.load_state_dict(self.model.state_dict())

    def to_tensor(self, obs: Observation) -> tuple[torch.Tensor, torch.Tensor]:
        extras = torch.from_numpy(obs.extras).unsqueeze(0).to(self.device)
        obs_tensor = torch.from_numpy(obs.data).unsqueeze(0).to(self.device)
        return obs_tensor, extras

    def compute_qvalues_and_hidden(self, observation: Observation):
        opt = self.opt
        hidden_res = []
        q_t_res = []
        with torch.no_grad():
            # TODO : Get agent_input : s_t = obs, messages (history), hidden (history), prev_action (history), agent_index ? 1 pass foreach agent ?
            self.episode.step_records.append(self.create_step_record())
            step_minus_1_id = -3
            step_id = -2
            step_plus_1_id = -1
            step_greater_0 = len(self.episode.step_records) > 2
            obs, extras = self.to_tensor(observation) 
            for i in range(self.opt.game_nagents):
                agent_idx = i
                comm = None
                if opt.comm_enabled:
                    comm = self.episode.step_records[step_id].comm.clone()
                    # comm_limited = self.game.get_comm_limited(step, agent.id)
                    # if comm_limited is not None:
                    #     comm_lim = torch.zeros(opt.bs, 1, opt.game_comm_bits)
                    #     for b in range(opt.bs):
                    #         if comm_limited[b].item() > 0:
                    #             comm_lim[b] = comm[b][comm_limited[b] - 1]
                    #     comm = comm_lim
                    # else:
                    comm[agent_idx].zero_()

                # Get prev action 
                prev_action = torch.tensor(0, dtype=torch.long).to(self.device)
                prev_message = torch.tensor(0, dtype=torch.long).to(self.device)
                if opt.model_action_aware:
                    if step_greater_0 and self.episode.step_records[step_minus_1_id].a_t[agent_idx] > 0:
                        prev_action= self.episode.step_records[step_minus_1_id].a_t[agent_idx]
                    if not opt.model_dial:
                        if step_greater_0 and self.episode.step_records[step_minus_1_id].a_comm_t[agent_idx] > 0:
                            prev_message = self.episode.step_records[step_minus_1_id].a_comm_t[agent_idx]
                    if not opt.model_dial:
                        prev_action = (prev_action, prev_message)

                agent_idx = torch.tensor(agent_idx, dtype=torch.long).to(self.device)

                agent_inputs = {
					'obs': obs,
                    'extras': extras,
					'messages': comm.to(self.device), # Messages
					'hidden': self.episode.step_records[step_id].hidden.to(self.device), # Hidden state # TODO : one hidden state per agent
					'prev_action': prev_action,
					'agent_index': agent_idx
				}
                # Compute model output (Q function + message bits)
                hidden_t, q_t = self.model.forward(**agent_inputs)
                hidden_res.append(hidden_t)
                q_t_res.append(q_t)
        return hidden_res, q_t_res

    def choose_action(self, observation: Observation) -> np.ndarray:
        opt = self.opt
        actions = []
        with torch.no_grad():
            # TODO : Get agent_input : s_t = obs, messages (history), hidden (history), prev_action (history), agent_index ? 1 pass foreach agent ?
            self.episode.step_records.append(self.create_step_record())
            step_minus_1_id = -3
            step_id = -2
            step_plus_1_id = -1
            step_greater_0 = len(self.episode.step_records) > 2
            obs, extras = self.to_tensor(observation) 
            for i in range(self.opt.game_nagents):
                agent_idx = i
                comm = None
                if opt.comm_enabled:
                    comm = self.episode.step_records[step_id].comm.clone()
                    # comm_limited = self.game.get_comm_limited(step, agent.id)
                    # if comm_limited is not None:
                    #     comm_lim = torch.zeros(opt.bs, 1, opt.game_comm_bits)
                    #     for b in range(opt.bs):
                    #         if comm_limited[b].item() > 0:
                    #             comm_lim[b] = comm[b][comm_limited[b] - 1]
                    #     comm = comm_lim
                    # else:
                    comm[agent_idx].zero_()

                # Get prev action 
                prev_action = torch.tensor(0, dtype=torch.long).to(self.device)
                prev_message = torch.tensor(0, dtype=torch.long).to(self.device)
                if opt.model_action_aware:
                    if step_greater_0 and self.episode.step_records[step_minus_1_id].a_t[agent_idx] > 0:
                        prev_action= self.episode.step_records[step_minus_1_id].a_t[agent_idx].to(self.device)
                    if not opt.model_dial:
                        if step_greater_0 and self.episode.step_records[step_minus_1_id].a_comm_t[agent_idx] > 0:
                            prev_message = self.episode.step_records[step_minus_1_id].a_comm_t[agent_idx].to(self.device)
                    if not opt.model_dial:
                        prev_action = (prev_action, prev_message)

                agent_idx = torch.tensor(agent_idx, dtype=torch.long).to(self.device)

                agent_inputs = {
					'obs': obs,
                    'extras': extras,
					'messages': comm.to(self.device), # Messages
					'hidden': self.episode.step_records[step_id].hidden.to(self.device), # Hidden state # TODO : one hidden state per agent
					'prev_action': prev_action,
					'agent_index': agent_idx
				}
                # Compute model output (Q function + message bits)
                hidden_t, q_t = self.model.forward(**agent_inputs)
                self.episode.step_records[step_plus_1_id].hidden = hidden_t

                # Choose next action and comm using eps-greedy selector
                (action, action_value), (comm_vector, comm_action, comm_value) = \
                    self.select_action_and_comm(observation, q_t, agent_idx)
                
                actions.append(action.cpu().item())

                # Store action + comm
                self.episode.step_records[step_id].a_t[agent_idx] = action
                self.episode.step_records[step_id].q_a_t[agent_idx] = action_value
                self.episode.step_records[step_plus_1_id].comm[agent_idx] = comm_vector
                if not opt.model_dial:
                    self.episode.step_records[step_id].a_comm_t[:, agent_idx] = comm_action
                    self.episode.step_records[step_id].q_comm_t[:, agent_idx] = comm_value

        return actions

    def value(self, obs: Observation) -> float:
        agent_values = self.compute_qvalues(obs).max(dim=-1).values
        return agent_values.mean(dim=-1).cpu().item()

    def compute_qvalues(self, obs: Observation) -> torch.Tensor:
        _, q = self.compute_qvalues_and_hidden(obs)
        return torch.stack(q)

    def set_testing(self):
        self.training_mode = False
        self.policy = self.test_policy

    def set_training(self):
        self.training_mode = True
        self.policy = self.train_policy

    def new_episode(self):
        # Clear the history 
        self.episode = self.create_episode()
        self.episode.step_records.append(self.create_step_record())
    
    #TODO : Save weights
    def save(self, to_directory: str):
        # os.makedirs(to_directory, exist_ok=True)
        # torch.save(self.qnetwork.state_dict(), f"{to_directory}/qnetwork.weights")
        # train_policy_path = os.path.join(to_directory, "train_policy")
        # test_policy_path = os.path.join(to_directory, "test_policy")
        # with open(train_policy_path, "wb") as f, open(test_policy_path, "wb") as g:
        #     pickle.dump(self.train_policy, f)
        #     pickle.dump(self.test_policy, g)
        pass
    #TODO : Load weights
    def load(self, from_directory: str):
        # self.qnetwork.load_state_dict(torch.load(f"{from_directory}/qnetwork.weights"))
        # train_policy_path = os.path.join(from_directory, "train_policy")
        # test_policy_path = os.path.join(from_directory, "test_policy")
        # with open(train_policy_path, "rb") as f, open(test_policy_path, "rb") as g:
        #     self.train_policy = pickle.load(f)
        #     self.test_policy = pickle.load(g)
        # self.policy = self.train_policy
        pass
    
    def randomize(self):
        self.model.randomize()
        self.model_target.randomize()
    
    def to(self, device: torch.device):
        self.model.to(device)
        self.model_target.to(device)
        self.device = device

        