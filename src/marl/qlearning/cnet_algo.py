import os
import pickle
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

from rlenv.models import Observation
from rlenv import Episode
from marl.utils import DotDic

from typing import Optional

from marl.models import RLAlgo, Policy, DRU
from marl.nn.model_bank import CNet

class EpisodeCommWrapper:
    def __init__(self):
        self.episodes = []
    
    ########## Save and Load content which is not in a Episode Batch ##########
    def create_episode(opt, bs=1):

        episode = DotDic({})
        episode.steps = torch.zeros(bs).int()
        episode.ended = torch.zeros(bs).int()
        episode.r = torch.zeros(bs, opt.game_nagents).float()
        episode.step_records = []

        return episode

    def create_step_record(opt, bs=1):
        record = DotDic({})
        record.s_t = None
        record.r_t = torch.zeros(bs, opt.game_nagents).squeeze(-1)
        record.terminal = torch.zeros(bs).squeeze(-1)

        record.agent_inputs = []

        # Track actions at time t per agent
        record.a_t = torch.zeros(bs, opt.game_nagents, dtype=torch.long).squeeze(-1)
        if not opt.model_dial:
            record.a_comm_t = torch.zeros(bs, opt.game_nagents, dtype=torch.long).squeeze(-1)

        # Track messages sent at time t per agent
        if opt.comm_enabled:
            comm_dtype = opt.model_dial and torch.float or torch.long
            comm_dtype = torch.float
            record.comm = torch.zeros(bs, opt.game_nagents, opt.game_comm_bits, dtype=comm_dtype).squeeze(-1)
            if opt.model_dial and opt.model_target:
                record.comm_target = record.comm.clone()

        # Track hidden state per time t per agent
        record.hidden = torch.zeros(opt.game_nagents, opt.model_rnn_layers, bs, opt.model_rnn_size).squeeze(2)
        record.hidden_target = torch.zeros(opt.game_nagents, opt.model_rnn_layers, bs, opt.model_rnn_size).squeeze(2)

        # Track Q(a_t) and Q(a_max_t) per agent
        record.q_a_t = torch.zeros(bs, opt.game_nagents).squeeze(-1)
        record.q_a_max_t = torch.zeros(bs, opt.game_nagents).squeeze(-1)

        # Track Q(m_t) and Q(m_max_t) per agent
        if not opt.model_dial:
            record.q_comm_t = torch.zeros(bs, opt.game_nagents).squeeze(-1)
            record.q_comm_max_t = torch.zeros(bs, opt.game_nagents).squeeze(-1)

        return record
    ###########################################################################

    def get_batch(self, opt):
        episode = EpisodeCommWrapper.create_episode(opt, len(self.episodes))
        for episode_id in range(len(self.episodes)):
            episode_from_agent = self.episodes[episode_id]
            episode.steps[episode_id] = episode_from_agent.steps
            episode.ended[episode_id] = episode_from_agent.ended
            episode.r[episode_id] = episode_from_agent.r
            for time_step in range(episode_from_agent.steps):
                record = episode_from_agent.step_records[time_step]
                episode.step_records.append(EpisodeCommWrapper.create_step_record(opt, len(self.episodes)))
                episode.step_records[time_step].r_t[episode_id] = record.r_t
                episode.step_records[time_step].terminal[episode_id] = record.terminal
                episode.step_records[time_step].a_t[episode_id] = record.a_t
                if not opt.model_dial:
                    episode.step_records[time_step].a_comm_t[episode_id] = record.a_comm_t
                if opt.comm_enabled:
                    episode.step_records[time_step].comm[episode_id] = record.comm
                episode.step_records[time_step].hidden[episode_id] = record.hidden
                episode.step_records[time_step].q_a_t[episode_id] = record.q_a_t
                episode.step_records[time_step].q_a_max_t[episode_id] = record.q_a_max_t
                if not opt.model_dial:
                    episode.step_records[time_step].q_comm_t[episode_id] = record.q_comm_t
                    episode.step_records[time_step].q_comm_max_t[episode_id] = record.q_comm_max_t
        
        return episode
    
    def add_episode(self, episode: Episode):
        self.episodes.append(episode)
    
    def clear(self):
        self.episodes = []


class CNetAlgo(RLAlgo):
    STEP_MINUS_1_ID = -3
    STEP_ID = -2
    STEP_PLUS_1_ID = -1
    
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
        
    def reset(self):
        self.model.reset_parameters()
        self.model_target.reset_parameters()
        self.episodes_seen = 0
    
    def to_tensor(self, obs: Observation) -> tuple[torch.Tensor, torch.Tensor]:
        extras = torch.from_numpy(obs.extras).unsqueeze(0).to(self.device)
        obs_tensor = torch.from_numpy(obs.data).unsqueeze(0).to(self.device)
        return obs_tensor, extras
    
    def select_action_and_comm(self, obs: Observation, q):
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
        action = self.policy.get_action(q.cpu().numpy()[:, :opt.game_action_space], obs.available_actions)
        action_value = q[torch.arange(q.size(0)), action]

        if not opt.model_dial:
            comm = self.policy.get_action(q[:, -opt.game_comm_bits:], [1 for _ in range(opt.game_comm_bits)])
            action_value = q[comm + opt.game_action_space]
            comm_vector[comm] = 1
        else:
            comm_vector = self.dru.forward(q[:, -opt.game_comm_bits:], self.training_mode)

        return (action, action_value), (comm_vector, comm_action, comm_value)

    def compute_qvalues_and_hidden(self, observation: Observation):
        opt = self.opt
        with torch.no_grad():
            
            step_greater_0 = len(self.episode.step_records) > 2
            obs, extras = self.to_tensor(observation) 
            comm = None
            if opt.comm_enabled:
                comm = self.episode.step_records[CNetAlgo.STEP_ID].comm.clone()

            # Get prev action 
            prev_action = torch.tensor(0, dtype=torch.long).to(self.device)
            prev_message = torch.tensor(0, dtype=torch.long).to(self.device)
            if opt.model_action_aware:
                if step_greater_0:
                    prev_action= self.episode.step_records[CNetAlgo.STEP_MINUS_1_ID].a_t.to(self.device)
                if not opt.model_dial:
                    if step_greater_0:
                        prev_message = self.episode.step_records[CNetAlgo.STEP_MINUS_1_ID].a_comm_t.to(self.device)
                if not opt.model_dial:
                    prev_action = (prev_action, prev_message)

            # agent_idx = torch.tensor(agent_idx, dtype=torch.long).to(self.device)

            agent_inputs = {
                'obs': obs,
                'extras': extras,
                'messages': comm.to(self.device), # Messages
                'hidden': self.episode.step_records[CNetAlgo.STEP_ID].hidden.to(self.device),
                'prev_action': prev_action
                #'agent_index': agent_idx
            }

            self.episode.step_records[CNetAlgo.STEP_ID].agent_inputs.append(agent_inputs)
                        
            # Compute model output (Q function + message bits)
            return self.model.forward(**agent_inputs)

    def choose_action(self, observation: Observation) -> np.ndarray:
        opt = self.opt
        with torch.no_grad():
            # TODO : Get agent_input : s_t = obs, messages (history), hidden (history), prev_action (history), agent_index ? 1 pass foreach agent ?
            self.episode.step_records.append(EpisodeCommWrapper.create_step_record(opt))           
            
            # Compute model output (Q function + message bits)
            hidden_t, q_t = self.compute_qvalues_and_hidden(observation)
            self.episode.step_records[CNetAlgo.STEP_PLUS_1_ID].hidden = hidden_t

            # Choose next action and comm using eps-greedy selector
            (action, action_value), (comm_vector, comm_action, comm_value) = \
                self.select_action_and_comm(observation, q_t)
            
            # Store action + comm
            self.episode.step_records[CNetAlgo.STEP_ID].a_t = torch.tensor(action).to(self.device)
            self.episode.step_records[CNetAlgo.STEP_ID].q_a_t = action_value
            self.episode.step_records[CNetAlgo.STEP_PLUS_1_ID].comm = comm_vector
            if not opt.model_dial:
                self.episode.step_records[CNetAlgo.STEP_ID].a_comm_t = comm_action
                self.episode.step_records[CNetAlgo.STEP_ID].q_comm_t = comm_value
        return action
    
    def fill_target_values(self, episode):
        opt = self.opt
        if opt.model_target:
            pass

    
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
        episode = self.fill_target_values(episode)
        loss = self.episode_loss(episode)
        loss.backward(retain_graph=not self.opt.model_know_share)
        clip_grad_norm_(parameters=self.model.get_params(), max_norm=10)
        self.optimizer.step()

        self.episodes_seen = self.episodes_seen + 1
        if self.episodes_seen % self.opt.step_target == 0:
            self.model_target.load_state_dict(self.model.state_dict())
    
    def get_episode(self):
        # The first step record is a dummy record
        self.episode.step_records.pop(0)
        return self.episode
    
    def get_step_record(self):	
        return self.episode.step_records[CNetAlgo.STEP_ID] # Get the step record of time t

    def value(self, obs: Observation) -> float:
        agent_values = self.compute_qvalues(obs).max(dim=-1).values
        return agent_values.mean(dim=-1).cpu().item()

    def compute_qvalues(self, obs: Observation) -> torch.Tensor:
        self.episode.step_records.append(EpisodeCommWrapper.create_step_record(self.opt))
        _, q = self.compute_qvalues_and_hidden(obs)
        self.episode.step_records.pop()
        q_values = q[:, :self.opt.game_action_space].max(dim=-1).values
        return q_values.mean(dim=-1)

    def set_testing(self):
        self.training_mode = False
        self.policy = self.test_policy

    def set_training(self):
        self.training_mode = True
        self.policy = self.train_policy

    def new_episode(self):
        # Clear the history 
        self.episode = EpisodeCommWrapper.create_episode(self.opt)
        self.episode.step_records.append(EpisodeCommWrapper.create_step_record(self.opt))
    
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

        