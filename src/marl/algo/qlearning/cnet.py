import os
import pickle
import copy
from types import SimpleNamespace
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

from marlenv.models import Observation
from marlenv import Episode
from marl.utils import DotDic

from typing import Optional
from marl.models import Policy, DRU
from marl.nn.model_bank import CNet


from ..algo import RLAlgo


class EpisodeCommWrapper:
    def __init__(self):
        self.episodes = []

    ########## Save and Load content which is not in a Episode Batch ##########
    @staticmethod
    def create_episode(opt: SimpleNamespace, bs=1):
        episode = DotDic({})
        episode.steps = torch.zeros(bs).int()
        episode.ended = torch.zeros(bs).int()
        episode.r = torch.zeros(bs, opt.game_nagents).float()
        episode.step_records = []

        return episode

    @staticmethod
    def create_step_record(opt: SimpleNamespace, device, bs=1):
        record = DotDic({})
        record.s_t = None
        record.r_t = torch.zeros(bs, opt.game_nagents).to(device)
        record.terminal = torch.zeros(bs).to(device)

        record.agent_inputs = {}

        # Track actions at time t per agent
        record.a_t = torch.zeros(bs, opt.game_nagents, dtype=torch.long).to(device)
        record.avail_a_t = torch.zeros(bs, opt.game_nagents, opt.game_action_space, dtype=torch.long).to(device)
        if not opt.model_dial:
            record.a_comm_t = torch.zeros(bs, opt.game_nagents, dtype=torch.long).to(device)

        # Track messages sent at time t per agent
        if opt.comm_enabled:
            comm_dtype = opt.model_dial and torch.float or torch.long
            comm_dtype = torch.float
            record.comm = torch.zeros(bs, opt.game_nagents, opt.game_comm_bits, dtype=comm_dtype).to(device)
            if opt.model_dial and opt.model_target:
                record.comm_target = record.comm.clone()

        # Track hidden state per time t per agent
        record.hidden = torch.zeros(opt.game_nagents, opt.model_rnn_layers, bs, opt.model_rnn_size).to(device)
        record.hidden_target = torch.zeros(opt.game_nagents, opt.model_rnn_layers, bs, opt.model_rnn_size).to(device)

        # Track Q(a_t) and Q(a_max_t) per agent
        record.q_a_t = torch.zeros(bs, opt.game_nagents).to(device)
        record.q_a_max_t = torch.zeros(bs, opt.game_nagents).to(device)

        # Track Q(m_t) and Q(m_max_t) per agent
        if not opt.model_dial:
            record.q_comm_t = torch.zeros(bs, opt.game_nagents).to(device)
            record.q_comm_max_t = torch.zeros(bs, opt.game_nagents).to(device)

        return record

    ###########################################################################
    @staticmethod
    def add_agent_inputs(episode, time_step, agent_inputs):
        if agent_inputs != {}:
            current_agent_input = episode.step_records[time_step].agent_inputs
            if current_agent_input == {}:
                obs = agent_inputs["obs"]
                extras = agent_inputs["extras"]
                messages = agent_inputs["messages"].unsqueeze(0)
                hidden = agent_inputs["hidden"]
                prev_action = agent_inputs["prev_action"].unsqueeze(0)

                episode.step_records[time_step].agent_inputs = {
                    "obs": obs,
                    "extras": extras,
                    "messages": messages,
                    "hidden": hidden,
                    "prev_action": prev_action,
                }
            else:  # Concatenate the agent inputs
                obs = torch.cat((current_agent_input["obs"], agent_inputs["obs"]), dim=0)
                extras = torch.cat((current_agent_input["extras"], agent_inputs["extras"]), dim=0)
                messages = torch.cat((current_agent_input["messages"], agent_inputs["messages"].unsqueeze(0)), dim=0)
                hidden = torch.cat((current_agent_input["hidden"], agent_inputs["hidden"]), dim=2)
                prev_action = torch.cat((current_agent_input["prev_action"], agent_inputs["prev_action"].unsqueeze(0)), dim=0)

                episode.step_records[time_step].agent_inputs = {
                    "obs": obs,
                    "extras": extras,
                    "messages": messages,
                    "hidden": hidden,
                    "prev_action": prev_action,
                }

        return episode

    def get_batch(self, opt, device):
        episode = EpisodeCommWrapper.create_episode(opt, len(self.episodes))
        ended_dict = {}
        for episode_id in range(len(self.episodes)):
            episode_from_agent = self.episodes[episode_id]
            episode.steps[episode_id] = episode_from_agent.steps
            episode.ended[episode_id] = episode_from_agent.ended
            episode.r[episode_id] = episode_from_agent.r
            for time_step in range(episode_from_agent.steps):
                if episode_id == 0:
                    bs = 0
                    for e in range(len(self.episodes)):
                        if self.episodes[e].steps > time_step:
                            bs = bs + 1
                        else:
                            if ended_dict.get(e) is None:
                                ended_dict[time_step] = [e]
                            else:
                                ended_dict[time_step].append(e)

                    episode.step_records.append(EpisodeCommWrapper.create_step_record(opt, device, bs))

                if ended_dict.get(time_step) is not None:
                    episode_id_in_record = episode_id - sum([1 for e in ended_dict[time_step] if e < episode_id])
                else:
                    episode_id_in_record = episode_id
                record = episode_from_agent.step_records[time_step]
                EpisodeCommWrapper.add_agent_inputs(episode, time_step, record.agent_inputs)
                episode.step_records[time_step].r_t[episode_id_in_record] = record.r_t
                episode.step_records[time_step].terminal[episode_id_in_record] = record.terminal
                episode.step_records[time_step].a_t[episode_id_in_record] = record.a_t
                episode.step_records[time_step].avail_a_t[episode_id_in_record] = torch.tensor(record.avail_a_t)
                if not opt.model_dial:
                    episode.step_records[time_step].a_comm_t[episode_id_in_record] = record.a_comm_t
                if opt.comm_enabled:
                    episode.step_records[time_step].comm[episode_id_in_record] = record.comm
                episode.step_records[time_step].hidden[:, :, episode_id_in_record, :] = record.hidden.squeeze(2)
                episode.step_records[time_step].q_a_t[episode_id_in_record] = record.q_a_t
                episode.step_records[time_step].q_a_max_t[episode_id_in_record] = record.q_a_max_t
                if not opt.model_dial:
                    episode.step_records[time_step].q_comm_t[episode_id_in_record] = record.q_comm_t
                    episode.step_records[time_step].q_comm_max_t[episode_id_in_record] = record.q_comm_max_t

        return episode

    def add_episode(self, episode: Episode):
        self.episodes.append(episode)

    def clear(self):
        self.episodes = []


class CNet(RLAlgo):
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

        self.optimizer = optim.RMSprop(params=model.get_params(), lr=opt.learningrate, momentum=opt.momentum)

        self.training_mode = False

    def reset(self):
        self.model.reset_parameters()
        self.model_target.reset_parameters()
        self.episodes_seen = 0

    def to_tensor(self, obs: Observation) -> tuple[torch.Tensor, torch.Tensor]:
        extras = torch.from_numpy(obs.extras).unsqueeze(0).to(self.device)
        obs_tensor = torch.from_numpy(obs.data).unsqueeze(0).to(self.device)
        return obs_tensor, extras

    def select_action_and_comm(self, q, avail_actions):
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
        bs = q.size(0)
        is_batched = bs > 1
        if not is_batched:
            q_action = q.squeeze(0)[:, : opt.game_action_space]
            q_message = q.squeeze(0)[:, -opt.game_comm_bits :]
        else:
            q_action = q[:, :, : opt.game_action_space]
            q_message = q[:, :, -opt.game_comm_bits :]

        if not is_batched:
            action = self.policy.get_action(q_action.cpu().numpy(), avail_actions)
        else:
            action = np.stack([self.policy.get_action(q_action[b, :].cpu().numpy(), avail_actions[b]) for b in range(bs)])
        if not is_batched:
            action_value = q_action[torch.arange(q_action.size(0)), action]
        else:
            # Reshape action to match the indexing requirements
            action_flat = action.flatten()  # Flatten action to a 1D tensor

            # Create indexing tensors for the first two dimensions
            index_0 = torch.arange(q_action.size(0)).unsqueeze(1).expand(-1, q_action.size(1)).reshape(-1)
            index_1 = torch.arange(q_action.size(1)).repeat(q_action.size(0))

            # Use advanced indexing to select values from q_action
            action_value = q_action[index_0, index_1, action_flat]

            # Reshape action_value to match the original shape of action
            action_value = action_value.view(action.shape)

        if not opt.model_dial:
            comm = self.policy.get_action(q_message.cpu().numpy(), np.ones((opt.game_comm_bits,)))
            if not is_batched:
                comm_value = q_message[torch.arange(q_message.size(0)), comm]
            else:
                comm_value = np.stack([q_message[b, :, comm[b]].cpu().numpy() for b in range(bs)])
            comm_vector[comm] = 1
        else:
            comm_vector = self.dru.forward(q_message, self.training_mode)

        return (action, action_value), (comm_vector, comm_action, comm_value)

    def compute_qvalues_and_hidden(self, observation: Observation):
        opt = self.opt
        step_greater_0 = len(self.episode.step_records) > 2
        obs, extras = self.to_tensor(observation)
        comm = None
        if opt.comm_enabled:
            comm = self.episode.step_records[CNet.STEP_ID].comm.clone()

        # Get prev action
        prev_action = torch.zeros(opt.game_nagents, dtype=torch.long).to(self.device)
        prev_message = torch.zeros(opt.game_nagents, dtype=torch.long).to(self.device)
        if opt.model_action_aware:
            if step_greater_0:
                prev_action = self.episode.step_records[CNet.STEP_MINUS_1_ID].a_t.to(self.device)
            if not opt.model_dial:
                if step_greater_0:
                    prev_message = self.episode.step_records[CNet.STEP_MINUS_1_ID].a_comm_t.to(self.device)
            if not opt.model_dial:
                prev_action = (prev_action, prev_message)
        # agent_idx = torch.tensor(agent_idx, dtype=torch.long).to(self.device)

        agent_inputs = {
            "obs": obs,
            "extras": extras,
            "messages": comm.to(self.device),  # type: ignore
            "hidden": self.episode.step_records[CNet.STEP_ID].hidden.to(self.device),
            "prev_action": prev_action,
            #'agent_index': agent_idx
        }
        self.episode.step_records[CNet.STEP_ID].avail_a_t = observation.available_actions
        self.episode.step_records[CNet.STEP_ID].agent_inputs = agent_inputs

        # Compute model output (Q function + message bits)
        return self.model.forward(**agent_inputs)

    def choose_action(self, observation: Observation) -> np.ndarray:
        opt = self.opt
        with torch.no_grad():
            # TODO : Get agent_input : s_t = obs, messages (history), hidden (history), prev_action (history), agent_index ? 1 pass foreach agent ?
            self.episode.step_records.append(EpisodeCommWrapper.create_step_record(opt, self.device))

            # Compute model output (Q function + message bits)
            hidden_t, q_t = self.compute_qvalues_and_hidden(observation)
            self.episode.step_records[CNet.STEP_PLUS_1_ID].hidden = hidden_t
            # Choose next action and comm using eps-greedy selector
            avail_actions = self.episode.step_records[CNet.STEP_ID].avail_a_t
            (action, action_value), (comm_vector, comm_action, comm_value) = self.select_action_and_comm(q_t, avail_actions)

            # Store action + comm
            self.episode.step_records[CNet.STEP_ID].a_t = torch.tensor(action).to(self.device)
            self.episode.step_records[CNet.STEP_ID].q_a_t = action_value
            self.episode.step_records[CNet.STEP_PLUS_1_ID].comm = comm_vector
            if not opt.model_dial:
                self.episode.step_records[CNet.STEP_ID].a_comm_t = comm_action
                self.episode.step_records[CNet.STEP_ID].q_comm_t = comm_value
        return action

    def fill_target_values(self, episode):
        opt = self.opt
        if opt.model_target:
            for step in range(opt.nsteps):
                agent_inputs = episode.step_records[step].agent_inputs
                comm_target = agent_inputs["messages"]
                if opt.comm_enabled and opt.model_dial:
                    comm_target = episode.step_records[step].comm_target.clone()

                agent_target_inputs = copy.copy(agent_inputs)
                agent_target_inputs["messages"] = comm_target
                agent_target_inputs["hidden"] = episode.step_records[step].hidden_target
                hidden_target_t, q_target_t = self.model_target.forward(**agent_target_inputs)

                ended_episode = []
                current_bs = agent_inputs["obs"].shape[0]
                for b in range(current_bs):
                    next_step_ended = episode.step_records[step].terminal[b].item()
                    if next_step_ended > 0:
                        ended_episode.append(b)

                if (step + 1) < opt.nsteps:
                    hidden_target_t_squeezed = hidden_target_t.squeeze()
                    hidden_target_t_remaining = None

                    for b in range(hidden_target_t_squeezed.shape[2]):
                        if b not in ended_episode:
                            if hidden_target_t_remaining is None:
                                hidden_target_t_remaining = hidden_target_t_squeezed[:, :, b].unsqueeze(2)
                            else:
                                hidden_target_t_remaining = torch.cat(
                                    (hidden_target_t_remaining, hidden_target_t_squeezed[:, :, b].unsqueeze(2)), dim=2
                                )

                    episode.step_records[step + 1].hidden_target = hidden_target_t_remaining

                # Choose next arg max action and comm
                avail_actions = episode.step_records[step].avail_a_t.squeeze(0).cpu().numpy()

                (action, action_value), (comm_vector, comm_action, comm_value) = self.select_action_and_comm(q_target_t, avail_actions)

                # save target actions, comm, and q_a_t, q_a_max_t
                episode.step_records[step].q_a_max_t = action_value
                if opt.model_dial:
                    comm_target_remaining = None
                    for b in range(comm_vector.shape[0]):
                        if b not in ended_episode:
                            if comm_target_remaining is None:
                                comm_target_remaining = comm_vector[b].unsqueeze(0)
                            else:
                                comm_target_remaining = torch.cat((comm_target_remaining, comm_vector[b].unsqueeze(0)), dim=0)
                    if (step + 1) < opt.nsteps:
                        episode.step_records[step + 1].comm_target = comm_target_remaining
                else:
                    episode.step_records[step].q_comm_max_t = comm_value

        return episode

    def episode_loss_2(self, episode):
        opt = self.opt
        total_loss = torch.zeros(opt.bs, device=self.device)
        ended_dict = {}

        for b in range(opt.bs):
            b_steps = episode.steps[b].item()
            for step in range(len(episode.step_records)):
                record = episode.step_records[step]
                if b == 0:
                    for e in range(opt.bs):
                        if episode.steps[e].item() <= step:
                            if ended_dict.get(e) is None:
                                ended_dict[step] = [e]
                            else:
                                ended_dict[step].append(e)
                if ended_dict.get(step) is not None:
                    b_id_in_record = b - sum([1 for e in ended_dict[step] if e < b])
                else:
                    b_id_in_record = b
                if ended_dict.get(step + 1) is not None:
                    b_plus_1_id_in_record = b - sum([1 for e in ended_dict[step + 1] if e < b])
                else:
                    b_plus_1_id_in_record = b + 1
                for i in range(opt.game_nagents):
                    td_action = torch.tensor(0.0, device=self.device)
                    td_comm = torch.tensor(0.0, device=self.device)

                    r_t = record.r_t[b_id_in_record][i]
                    q_a_t = record.q_a_t[b_id_in_record][i]
                    q_comm_t = 0

                    if record.a_t[b_id_in_record][i].item() > 0:
                        if record.terminal[b_id_in_record].item() > 0 or step >= b_steps - 1:
                            td_action = r_t - q_a_t
                        else:
                            next_record = episode.step_records[step + 1]
                            q_next_max = next_record.q_a_max_t[b_plus_1_id_in_record][i]
                            if not opt.model_dial and opt.model_avg_q:
                                q_next_max = (q_next_max + next_record.q_comm_max_t[b_plus_1_id_in_record][i]) / 2.0
                            td_action = r_t + opt.gamma * q_next_max - q_a_t

                    if not opt.model_dial and record.a_comm_t[b_id_in_record][i].item() > 0:
                        q_comm_t = record.q_comm_t[b_id_in_record][i]
                        if record.terminal[b_id_in_record].item() > 0 or step >= b_steps - 1:
                            td_comm = r_t - q_comm_t
                        else:
                            next_record = episode.step_records[step + 1]
                            q_next_max = next_record.q_comm_max_t[b_plus_1_id_in_record][i]
                            if opt.model_avg_q:
                                q_next_max = (q_next_max + next_record.q_a_max_t[b_plus_1_id_in_record][i]) / 2.0
                            td_comm = r_t + opt.gamma * q_next_max - q_comm_t

                    if not opt.model_dial:
                        loss_t = td_action**2 + td_comm**2
                    else:
                        loss_t = td_action**2
                    total_loss[b] = total_loss[b] + loss_t

        loss = total_loss.mean()
        return loss

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
                        if record.terminal[b].item() > 0 or step >= b_steps - 1:
                            td_action = r_t - q_a_t
                        else:
                            next_record = episode.step_records[step + 1]
                            q_next_max = next_record.q_a_max_t[i]
                            if not opt.model_dial and opt.model_avg_q:
                                q_next_max = (q_next_max + next_record.q_comm_max_t[b][i]) / 2.0
                            td_action = r_t + opt.gamma * q_next_max - q_a_t

                    if not opt.model_dial and record.a_comm_t[b][i].item() > 0:
                        q_comm_t = record.q_comm_t[b][i]
                        if record.terminal[b].item() > 0 or step >= b_steps - 1:
                            td_comm = r_t - q_comm_t
                        else:
                            next_record = episode.step_records[step + 1]
                            q_next_max = next_record.q_comm_max_t[b][i]
                            if opt.model_avg_q:
                                q_next_max = (q_next_max + next_record.q_a_max_t[b][i]) / 2.0
                            td_comm = r_t + opt.gamma * q_next_max - q_comm_t

                    if not opt.model_dial:
                        loss_t = td_action**2 + td_comm**2
                    else:
                        loss_t = td_action**2
                    total_loss[b] = total_loss[b] + loss_t
        loss = total_loss.sum()
        loss = loss / (self.opt.bs * self.opt.game_nagents)
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

        # TODO returns logs

    def get_episode(self):
        # The first step record is a dummy record
        self.episode.step_records.pop()
        return self.episode

    def get_step_record(self):
        return self.episode.step_records[CNet.STEP_ID]  # Get the step record of time t

    def clear_last_record(self):  # When the episode is done
        return 0

    def value(self, obs: Observation) -> float:
        agent_values = self.compute_qvalues(obs).max(dim=-1).values
        return agent_values.mean(dim=-1).cpu().item()

    def compute_qvalues(self, obs: Observation) -> torch.Tensor:
        self.episode.step_records.append(EpisodeCommWrapper.create_step_record(self.opt, self.device))
        _, q = self.compute_qvalues_and_hidden(obs)
        self.episode.step_records.pop()
        q_values = q[:, : self.opt.game_action_space].max(dim=-1).values
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
        self.episode.step_records.append(EpisodeCommWrapper.create_step_record(self.opt, self.device))

    # TODO : Save weights
    def save(self, to_directory: str):
        # os.makedirs(to_directory, exist_ok=True)
        # torch.save(self.qnetwork.state_dict(), f"{to_directory}/qnetwork.weights")
        # train_policy_path = os.path.join(to_directory, "train_policy")
        # test_policy_path = os.path.join(to_directory, "test_policy")
        # with open(train_policy_path, "wb") as f, open(test_policy_path, "wb") as g:
        #     pickle.dump(self.train_policy, f)
        #     pickle.dump(self.test_policy, g)
        pass

    # TODO : Load weights
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
