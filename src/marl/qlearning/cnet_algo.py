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

        for p in self.model_target.parameters():
            p.requires_grad = False
        
        self.episodes_seen = 0
        self.dru = DRU(opt.game_comm_sigma, opt.model_comm_narrow, opt.game_comm_hard)

        self.optimizer = optim.RMSprop(
            params=model.get_params(), lr=opt.learningrate, momentum=opt.momentum)
        
        # TODO : Add Container with the history of select_action_and_comm + addition value not present in EpisodeBatch (to determine)
        self.training_mode = False
        self.step = 0

               
        # self.train_policy = train_policy
        # if test_policy is None:
        #     test_policy = self.train_policy
        # self.test_policy = test_policy
        # self.policy = self.train_policy
    
    ########## Save and Load content which is not in a Episode Batch ##########
    def create_episode(self, bs=1):
        opt = self.opt

        episode = DotDic({})
        episode.steps = torch.zeros(bs).int().to(self.device)
        episode.ended = torch.zeros(bs).int().to(self.device)
        episode.r = torch.zeros(bs, opt.game_nagents).float().to(self.device)
        episode.step_records = []

        return episode

    def create_step_record(self, bs=1):
        opt = self.opt
        record = DotDic({})
        record.s_t = None
        record.r_t = torch.zeros(bs, opt.game_nagents).to(self.device)
        record.terminal = torch.zeros(bs).to(self.device)

        record.agent_inputs = []

        # Track actions at time t per agent
        record.a_t = torch.zeros(bs, opt.game_nagents, dtype=torch.long)
        if not opt.model_dial:
            record.a_comm_t = torch.zeros(bs, opt.game_nagents, dtype=torch.long)

        # Track messages sent at time t per agent
        if opt.comm_enabled:
            comm_dtype = opt.model_dial and torch.float or torch.long
            comm_dtype = torch.float
            record.comm = torch.zeros(bs, opt.game_nagents, opt.game_comm_bits, dtype=comm_dtype)
            if opt.model_dial and opt.model_target:
                record.comm_target = record.comm.clone()

        # Track hidden state per time t per agent
        record.hidden = torch.zeros(opt.game_nagents, opt.model_rnn_layers, bs, opt.model_rnn_size)
        record.hidden_target = torch.zeros(opt.game_nagents, opt.model_rnn_layers, bs, opt.model_rnn_size)

        # Track Q(a_t) and Q(a_max_t) per agent
        record.q_a_t = torch.zeros(bs, opt.game_nagents)
        record.q_a_max_t = torch.zeros(bs, opt.game_nagents)

        # Track Q(m_t) and Q(m_max_t) per agent
        if not opt.model_dial:
            record.q_comm_t = torch.zeros(bs, opt.game_nagents)
            record.q_comm_max_t = torch.zeros(bs, opt.game_nagents)

        return record
    ###########################################################################
        
    def reset(self):
        self.model.reset_parameters()
        self.model_target.reset_parameters()
        self.episodes_seen = 0
    
    # def _eps_flip(self, eps):
    #     # Sample Bernoulli with P(True) = eps
    #     return np.random.rand(self.opt.bs) < eps
    
    def get_action_comm_range(obs: Observation):
        # Get the range of actions and communications
        # foreach batch : if action available : add range to the result (1, action_space) else (1, 1)
        pass

    
    def select_action_and_comm(self, q, eps=0, target=False):
        # eps-Greedy action selector
        opt = self.opt
        action_range, comm_range = self.get_action_comm_range() # TODO according to the game
        action = torch.zeros(opt.bs, dtype=torch.long)
        action_value = torch.zeros(opt.bs)
        comm_action = torch.zeros(opt.bs).int()
        comm_vector = torch.zeros(opt.bs, opt.game_comm_bits)
        comm_value = None
        if not opt.model_dial:
            comm_value = torch.zeros(opt.bs)

        # should_select_random_comm = None
        # should_select_random_a = self._eps_flip(eps) # TODO : Use Policy 
        # if not opt.model_dial:
        #     should_select_random_comm = self._eps_flip(eps)

        # Get action + comm

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
    
    def choose_action(self, obs: Observation) -> np.ndarray:
        opt = self.opt
        with torch.no_grad():
            # TODO : Get agent_input : s_t = obs, messages (history), hidden (history), prev_action (history), agent_index ? 1 pass foreach agent ?
            self.episode.step_records.append(self.create_step_record(opt.bs))
            self.step += 1
            obs, extras = self.to_tensor(obs) 
            for i in range(self.opt.game_nagents):
                agent_id = i
                # TODO : improve for batch
                # messages = self.episode.step_records[self.step-1].comm.to(self.device) # Comm from t is stored on t+1
                # prev_actions = self.episode.step_records[self.step-1].a_t[agent_id].to(self.device) # Action from t is stored on t
                # hidden = self.episode.step_records[self.step-1].hidden.to(self.device) # Hidden from t is stored on t
                # Get received messages per agent per batch
                agent_idx = i
                comm = None
                if opt.comm_enabled:
                    comm = self.episode.step_records[self.step].comm.clone()
                    # comm_limited = self.game.get_comm_limited(step, agent.id)
                    # if comm_limited is not None:
                    #     comm_lim = torch.zeros(opt.bs, 1, opt.game_comm_bits)
                    #     for b in range(opt.bs):
                    #         if comm_limited[b].item() > 0:
                    #             comm_lim[b] = comm[b][comm_limited[b] - 1]
                    #     comm = comm_lim
                    # else:
                    comm[:, agent_idx].zero_()

                # Get prev action per batch
                prev_action = None
                if opt.model_action_aware:
                    prev_action = torch.ones(opt.bs, dtype=torch.long)
                    if not opt.model_dial:
                        prev_message = torch.ones(opt.bs, dtype=torch.long)
                    for b in range(opt.bs):
                        if self.step > 0 and self.episode.step_records[self.step - 1].a_t[b, agent_idx] > 0:
                            prev_action[b] = self.episode.step_records[self.step - 1].a_t[b, agent_idx]
                        if not opt.model_dial:
                            if self.step > 0 and self.episode.step_records[self.step - 1].a_comm_t[b, agent_idx] > 0:
                                prev_message[b] = self.episode.step_records[self.step - 1].a_comm_t[b, agent_idx]
                    if not opt.model_dial:
                        prev_action = (prev_action, prev_message)

                # Batch agent index for input into model
                batch_agent_index = torch.zeros(opt.bs, dtype=torch.long).fill_(agent_idx)

                agent_inputs = {
					'obs': obs,
                    'extras': extras,
					'messages': comm.to(self.device), # Messages
					'hidden': self.episode.step_records[self.step].hidden[agent_idx, :].to(self.device), # Hidden state
					'prev_action': prev_action.to(self.device),
					'agent_index': batch_agent_index.to(self.device)
				}
                # Compute model output (Q function + message bits)
                hidden_t, q_t = self.model.forward(**agent_inputs)
                self.episode.step_records[self.step+1].hidden[agent_id] = hidden_t

                # Choose next action and comm using eps-greedy selector
                (action, action_value), (comm_vector, comm_action, comm_value) = \
                    self.select_action_and_comm(q_t)

            qvalues = self.compute_qvalues(obs)
            # TODO : Store action and comm in the history : for qvalue t+1 : messages (history), hidden (history), prev_action (history)                                           for update ?        
        qvalues = qvalues.cpu().numpy()
        
        return self.policy.get_action(qvalues, obs.available_actions) # TODO : Replace by select_action_and_comm

    def value(self, obs: Observation) -> float:
        #return self.model.value(obs).item()
        return 0

    def compute_qvalues(self, obs: Observation) -> torch.Tensor:
        message = self._encode_message(obs)
        q_input = self._add_message_to_observation(obs, message)
        return self.qnetwork.qvalues(q_input)

    def set_testing(self):
        self.training_mode = False

    def set_training(self):
        self.training_mode = True

    def new_episode(self):
        # Clear the history 
        self.episode = self.create_episode(self.opt.bs)
        self.episode.step_records.append(self.create_step_record(self.opt.bs))
    
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

        