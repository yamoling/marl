import os

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pathlib
import numpy as np

from typing import Optional, Dict, Union

from marl.models.batch import Batch
from marl.xmarl.distilers.utils import flatten_observation, get_agent_pos, get_fixed_features, abstract_observation
from marlenv.models import Episode

EPS = 1e-6

class InnerNode(nn.Module):
    """SoftDecisionTree: a class representing an Inner Node in a SDT.
    Will recursively build the tree, by building its children.
    Almost copy pasted from: https://github.com/kimhc6028/soft-decision-tree/"""
    def __init__(self,
                 depth: int,
                 input_shape: tuple[int, ...],
                 output_shape: tuple[int, ...],
                 max_depth: int,
                 lmbda: float,
                 device: Optional[torch.device],
                 ):
        super(InnerNode, self).__init__()
        # Fix arguments
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.max_depth = max_depth
        self.lmbda = lmbda* 2 ** (-depth) # Lambda decays with depth
        self.device = device
        # Modules
        self.fc = nn.Linear(self.input_shape, 1)
        self.beta = nn.Parameter(torch.tensor(10.0)) # Fixed at one, to keep in check
        # Tree specific
        self.leaf = False
        self.prob = None
        self.leaf_accumulator = []
        self.penalties = []
        self.build_child(depth)

    def build_child(self, depth):
        """Builds the current node's children w.r.t depth, if depth = max_depth, builds nodes."""
        if depth < self.max_depth:
            self.left = InnerNode(depth+1, self.input_shape, self.output_shape, self.max_depth, self.lmbda, self.device)
            self.right = InnerNode(depth+1, self.input_shape, self.output_shape, self.max_depth, self.lmbda, self.device)
        else :
            self.left = LeafNode(depth+1, self.output_shape, self.device)
            self.right = LeafNode(depth+1, self.output_shape, self.device)

    def reset(self):
        """Resets the penalties and leaf accumulator and calls reset for children"""
        self.leaf_accumulator = []
        self.penalties = []
        self.left.reset()
        self.right.reset()

    def forward(self, x):
        """Applies the sigmoid function on the product of beta and the linear transformation"""
        return(torch.sigmoid(self.beta*self.fc(x)))
    
    def select_next(self, x):
        """Select the next node to take based on the probability"""
        prob = self.forward(x)
        if prob < 0.5:
            return(self.left, prob)
        else:
            return(self.right, prob)

    def cal_prob(self, x, path_prob):
        self.prob = self.forward(x) #probability of selecting right node
        if torch.any(torch.isnan(self.prob)) or torch.any(torch.isinf(self.prob)):
            print("prob: NaN or Inf detected!")
        self.path_prob = path_prob
        left_leaf_accumulator = self.left.cal_prob(x, path_prob * (1-self.prob))
        right_leaf_accumulator = self.right.cal_prob(x, path_prob * self.prob)
        self.leaf_accumulator.extend(left_leaf_accumulator)
        self.leaf_accumulator.extend(right_leaf_accumulator)
        return(self.leaf_accumulator)

    def get_penalty(self):
        """Recursively computes the penalty of each node"""
        penalty = (torch.sum(self.prob * self.path_prob) / torch.sum(self.path_prob).clamp(min=EPS), self.lmbda)
        if torch.any(torch.isnan(penalty[0])) or torch.any(torch.isinf(penalty[0])):
            print("penalty[0]: NaN or Inf detected!")
        if not self.left.leaf:
            left_penalty = self.left.get_penalty()
            right_penalty = self.right.get_penalty()
            self.penalties.append(penalty)
            self.penalties.extend(left_penalty)
            self.penalties.extend(right_penalty)
        return(self.penalties)
    
    def getfilters(self):
        """Returns the learned filter of the node, The added bias is ignored, supposed to be negligible"""
        return next(iter(self.fc.parameters())).data[0].cpu().numpy()*self.beta.data.cpu().numpy()

class LeafNode(nn.Module):
    """SoftDecisionTree: a class representing a Leaf Node in a SDT
    Almost copy pasted from: https://github.com/kimhc6028/soft-decision-tree/"""
    def __init__(self,
                 depth: int,
                 output_shape: tuple[int, ...],
                 device: Optional[torch.device] = None
                 ):
        super(LeafNode, self).__init__()
        self.depth = depth
        self.output_shape = output_shape
        self.device = device

        self.param = nn.Parameter(torch.randn(self.output_shape))
        self.softmax = nn.Softmax(dim=-1)

        self.leaf = True

    def forward(self):
        return(self.softmax(self.param.view(1,-1)))

    def reset(self):
        pass

    def cal_prob(self, x, path_prob):
        Q = self.forward()
        Q = Q.expand((path_prob.size(0), self.output_shape))
        return([[path_prob, Q]])

class SoftDecisionTree[B: Batch](nn.Module):
    """SoftDecisionTree: a class representing a Soft Decision Tree model distilled from a dnn
    Almost copy pasted from: https://github.com/kimhc6028/soft-decision-tree/"""
    batch_size: int
    input_shape: int
    logdir: pathlib.Path
    output_shape: int
    max_depth: int
    lr: float
    lmbda: float
    momentum: float
    device: Optional[torch.device]
    seed: int
    log_interval: int # not sure I need, enforced/done by DQN?

    n_agent: int
    agent_id: int
    extras: bool
    abstract: bool

    def __init__(self, 
                 input_shape: int,
                 output_shape: int,
                 logdir: str,
                 extras: bool,
                 abstract: bool,
                 n_agent: int = 1,
                 max_depth: int = 4, 
                 seed: int = 0,
                 bs: int = 64,
                 lr: Optional[float] = 0.01,
                 lmbda: Optional[float] = 0.01,
                 momentum: Optional[float] = 0.01,
                 device: Optional[torch.device] = torch.device('cpu'),
                 agent_id: Optional[int] = None
                 ):
        super(SoftDecisionTree, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_agent = n_agent
        self.agent_id = agent_id
        self.extras = extras
        self.abstract = abstract
        if self.agent_id is None: self.logdir = pathlib.Path(logdir, "distil")
        else: self.logdir = pathlib.Path(logdir, "distil", "individual_sdt_distil")

        self.max_depth = max_depth
        self.seed = seed

        self.batch_size = bs
        self.log_interval = bs/8
        self.lr = lr
        self.lmbda = lmbda
        self.momentum = momentum
        self.device = device

        self.root = InnerNode(1, self.input_shape, self.output_shape, self.max_depth, self.lmbda, self.device)

        self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)

        self.test_acc = []
        self.define_extras(self.batch_size)
        self.best_accuracy = 0.0
    
    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def define_extras(self, batch_size):
        self.path_prob_init = torch.ones(batch_size, 1, device=self.device)
     
    def cal_loss(self, x, y):
        leaf_accumulator = self.root.cal_prob(x, self.path_prob_init)
        loss = 0.
        max_prob = [-1. for _ in range(self.batch_size)]
        max_Q = torch.zeros((self.batch_size,self.output_shape))
        for (path_prob, Q) in leaf_accumulator:
            if torch.any(torch.isnan(Q)) or torch.any(torch.isinf(Q)):
                print("Q: NaN or Inf detected!")
            log_Q = torch.log(Q.clamp(min=EPS))
            TQ = torch.bmm(y.view(self.batch_size, 1, self.output_shape), log_Q.view(self.batch_size, self.output_shape, 1)).view(-1,1)
            if torch.any(torch.isnan(TQ)) or torch.any(torch.isinf(TQ)):
                print("TQ: NaN or Inf detected!")
            loss += path_prob * TQ
            path_prob_numpy = path_prob.cpu().data.numpy().reshape(-1)
            for i in range(self.batch_size):
                if max_prob[i] < path_prob_numpy[i]:
                    max_prob[i] = path_prob_numpy[i]
                    max_Q[i] = Q[i]

        loss = loss.mean()
        # Regalurization penalty
        penalties = self.root.get_penalty()
        C = 0.
        for (penalty, lmbda) in penalties:
            penalty = penalty.clamp(EPS, 1 - EPS) # Safeguard to not have penalty = 1 or 0
            C -= lmbda * 0.5 *(torch.log(penalty) + torch.log(1-penalty))
            if torch.any(torch.isnan(C)) or torch.any(torch.isinf(C)):
                print("C: NaN or Inf detected!")
            if torch.any(torch.isnan(penalty)) or torch.any(torch.isinf(penalty)):
                print("penalty: NaN or Inf detected!")
        
        self.root.reset() ##reset all stacked calculation
        if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
            print("loss: NaN or Inf detected!")

        return(-loss + C, max_Q) # -log(loss) in paper, loss already in logspace here


    def train_(self, train_data, train_targets, epoch=0):
        """While the model outputs a distribution over actions, train_data should have the one-hot encoded action choice"""
        if self.agent_id is not None: print(f"Training for agent {self.agent_id}")
        self.train()
        self.define_extras(self.batch_size)
        train_log = []
        for batch_idx  in range (len(train_data)):
            correct = 0
            target = torch.Tensor(train_targets[batch_idx], device=self.device)
            data = torch.Tensor(train_data[batch_idx], device=self.device)

            if self.agent_id is None:   # Ugly solution for individual distil
                batch_size = target.shape[0]
                if not batch_size == self.batch_size: #because we have to initialize parameters for batch_size, tensor not matches with batch size cannot be trained
                    self.define_extras(batch_size)

            self.optimizer.zero_grad()

            loss, output = self.cal_loss(data, target)
            #loss.backward(retain_variables=True)
            loss.backward()
            self.optimizer.step()
            pred = output.data.max(-1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.max(-1)[1]).cpu().sum() # -1 because action on last dim
            accuracy = 100. * correct / len(data)

            if batch_idx % self.log_interval == 0:
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.4f}%)'.format(
                #     epoch, batch_idx, len(train_data),
                #     100. * batch_idx / len(train_data), loss.item(),
                #     correct, len(data),
                #     accuracy))
                train_log.append((accuracy, loss.item()))
        train_log = np.array(train_log)
        print(f"Train accuracy for epoch {epoch}: {np.mean(train_log[:,0])}\n ")
        return train_log

    def test_(self, train_data, train_targets, epoch=0, test=False):
        if self.agent_id is not None: print(f"\nTesting for agent {self.agent_id}")
        self.eval()

        self.define_extras(self.batch_size)
        test_loss = 0
        correct = 0
        batch_nb = len(train_data)
        predictions = []
        for batch_idx  in range (batch_nb):

            target = torch.Tensor(train_targets[batch_idx], device=self.device)
            data = torch.Tensor(train_data[batch_idx], device=self.device)

            if self.agent_id is None:   # Ugly solution for individual distil
                batch_size = target.shape[0]
                if not batch_size == self.batch_size: #because we have to initialize parameters for batch_size, tensor not matches with batch size cannot be trained
                    self.define_extras(batch_size)

            _, output = self.cal_loss(data, target)
            pred = output.data.max(-1)[1] # get the index of the max log-probability
            predictions.append(pred.data.cpu())
            correct += pred.eq(target.data.max(-1)[1]).cpu().sum()

        total_data = batch_nb*self.batch_size
        accuracy = 100.*correct/total_data
        print(f"{"Validation" if not test else "Test"} set Accuracy: {correct}/{total_data} ({accuracy:.4f}%)\n")
        self.test_acc.append(accuracy)

        if accuracy > self.best_accuracy:
            self.save_best()
            self.best_accuracy = accuracy
        return accuracy, np.array(predictions).flatten()

    def greedy_trace(self, obs) -> tuple[LeafNode, list[InnerNode]]:
        """Greedy deterministic path through the SDT based on sigmoid threshold 0.5 and returns each filter."""
        path = []
        node = self.root
        while not node.leaf:
            prob = node.forward(torch.Tensor(obs, device=self.device)).item()
            path.append(node.getfilters())
            node = node.right if prob >= 0.5 else node.left
        return node, path  # node is leaf, use to get output
    
    def collect_leaves(self, node, leaves, outputs, parent_child, lf_c = 0):
        if node.leaf:
            leaves.append(node)
            outputs[lf_c] = node.forward().data.numpy().squeeze() # Is output in form (1,n_act), squeeze to (n_act,)
            lf_c += 1
        else:
            parent_child[node.left] = node
            parent_child[node.right] = node
            lf_c = self.collect_leaves(node.left, leaves, outputs, parent_child, lf_c)
            lf_c = self.collect_leaves(node.right, leaves, outputs, parent_child, lf_c)
        return lf_c

    def associate_action_leaf(self, action_idx: np.ndarray[int], leaves: list[LeafNode], outputs: np.ndarray[np.ndarray[float]]):
        """Traverse all leaves in the tree to find the one closest associating to a specific action."""
        best_leaves = []
        for i in range(len(action_idx)):
            a_idx = action_idx[i]
            best_leaf_idx = np.argmax(outputs[:, a_idx]).item()
            best_leaves.append(leaves[best_leaf_idx])
        return best_leaves
    
    def backtrack_leaf(self, leaf: LeafNode, parent_child: dict[Union[LeafNode|InnerNode],InnerNode]):
        """Given a leaf node, traces back to the root and returns each filter.
        The filters order is reversed to be in top to bottom order."""
        path = []
        node = leaf
        while node in parent_child:
            node = parent_child[node]
            path.append(node.getfilters())
        return np.array(path)[::-1]

    def distil_episode(self, episode: Episode, backwards: bool = False):
        """ Distils an episode, either by forward or backward pass, compounded or individually.
        Returns distilled episode board filters, distilled episode extras filters, distilled actions and agent positions.
        If abstracted obs will also return abstracted obs and extras.
        """
        if self.agent_id is None:
            return self.distil_episode_comp(episode, backwards)
        else:
            return self.distil_episode_ind(episode, backwards)

    def distil_episode_ind(self, episode: Episode, backwards: bool = False):
        obs_w = episode.observation_shape[-1]
        obs_h = episode.observation_shape[-2]

        paths_taken = [] # Will be filled with paths
        if self.abstract:
            abs_obs = []
            fix_ft = get_fixed_features(episode.all_observations[0])
            agent_pos = get_agent_pos(np.array(episode.all_observations))
        else:
            agent_pos = get_agent_pos(np.array(episode.all_observations))[:,self.agent_id]

        ep_acts = np.array(episode.actions)[:,self.agent_id]

        if not backwards:
            pred_actions = np.zeros((episode.episode_len,episode.n_actions),dtype=float)
            for i in range(episode.episode_len):
                if not self.abstract: 
                    f_obs = flatten_observation(episode.all_observations[i], episode.n_agents, 1) # Gonna flatten for all, inefficient in individual
                else: 
                    f_obs = abstract_observation(episode.all_observations[i], fix_ft, agent_pos[i])
                    abs_obs.append(f_obs[self.agent_id])
                if self.extras: 
                    if not self.abstract:
                        leaf, path = self.greedy_trace(np.concat([f_obs[self.agent_id].reshape(-1), episode.all_extras[i][self.agent_id], agent_pos[i]]))
                    else:
                        leaf, path = self.greedy_trace(f_obs[self.agent_id]+episode.all_extras[i][self.agent_id].tolist())
                else: 
                    if not self.abstract:
                        leaf, path = self.greedy_trace(f_obs[self.agent_id].reshape(-1))
                    else: 
                        leaf, path = self.greedy_trace(f_obs[self.agent_id])
                pred_actions[i] = leaf.forward().data.numpy().flatten()
                paths_taken.append(path)
            og_acts = np.zeros_like(pred_actions, dtype=int)
            np.put_along_axis(og_acts, ep_acts[..., np.newaxis], 1, axis=-1)
            actions_comp = np.stack([pred_actions, og_acts], axis=-2)
        else:   # Backwards
            leaves = []
            parent_child = {}
            outputs = np.zeros((2**self.max_depth,episode.n_actions), dtype=float)
            self.collect_leaves(self.root, leaves, outputs, parent_child) 
            for act_idx in ep_acts:
                leaf = self.associate_action_leaf([act_idx], leaves, outputs)[0] # Get best leaf (only one since individual)
                paths_taken.append(self.backtrack_leaf(leaf, parent_child)) # Right now stores filters of path not actual path
            actions_comp = np.zeros((episode.episode_len, episode.n_actions), dtype=int)
            np.put_along_axis(actions_comp, ep_acts[..., np.newaxis], 1, axis=1)

        paths_taken = np.array(paths_taken)
        # If applicable separate extras from board
        if not self.abstract:
            if self.extras:
                obs_f = paths_taken[:, :, :obs_w*obs_h].reshape(episode.episode_len, self.max_depth, obs_h, obs_w)
                extras_f = paths_taken[: , :, obs_w*obs_h:]
            else: 
                obs_f = paths_taken.reshape(episode.episode_len, self.max_depth, obs_h, obs_w)
                extras_f = None
        else:
            obs_f = paths_taken
        if not self.abstract: return obs_f, extras_f, actions_comp, agent_pos, None, None, None
        else: return obs_f, None, actions_comp, agent_pos[:,self.agent_id], abs_obs, np.array(episode.all_extras)[:,self.agent_id], fix_ft

    def distil_episode_comp(self, episode: Episode, backwards: bool = False):
        obs_w = episode.observation_shape[-1]
        obs_h = episode.observation_shape[-2]
        extras = obs_w*obs_h < self.input_shape

        paths_taken = [] # Will be filled with paths

        ep_acts = np.array(episode.actions)

        agent_pos = get_agent_pos(np.array(episode.all_observations))

        if not backwards:
            pred_actions = np.zeros((episode.episode_len,episode.n_agents,episode.n_actions),dtype=float)
            for i in range(episode.episode_len):
                f_obs = flatten_observation(episode.all_observations[i], episode.n_agents, 1)
                paths = []
                leaves = []
                for ag in range(episode.n_agents):
                    if extras: 
                        leaf, path = self.greedy_trace(np.concat([f_obs[ag].reshape(-1), episode.all_extras[i][ag], agent_pos[i][ag]]))
                        leaves.append(leaf)
                        paths.append(path)
                    else: 
                        leaf, path = self.greedy_trace(f_obs[ag].reshape(-1))
                        leaves.append(leaf)
                        paths.append(path)
                    pred_actions[i,ag] = leaf.forward().data.numpy().flatten()
                paths_taken.append(paths)
            og_acts = np.zeros_like(pred_actions, dtype=int)
            np.put_along_axis(og_acts, ep_acts[..., np.newaxis], 1, axis=-1)
            actions_comp = np.stack([pred_actions, og_acts], axis=-2)
        else:
            leaves = []
            parent_child = {}
            # TODO: get abstract data if abstract
            outputs = np.zeros((2**self.max_depth,episode.n_actions), dtype=float)
            self.collect_leaves(self.root, leaves, outputs, parent_child)
            for act in ep_acts:
                best_leaves = self.associate_action_leaf(act, leaves, outputs) # Get best leaf per agent
                paths_taken.append([self.backtrack_leaf(leaf, parent_child) for leaf in best_leaves]) # Right now stores filters of path not actual path
            actions_comp = np.zeros((episode.episode_len, episode.n_agents, episode.n_actions), dtype=int)
            np.put_along_axis(actions_comp, ep_acts[..., np.newaxis], 1, axis=2)

        paths_taken = np.array(paths_taken)
        # If applicable separate extras from board
        if extras:
            obs_f = paths_taken[:, :, :, :obs_w*obs_h].reshape(episode.episode_len, episode.n_agents, self.max_depth, obs_h, obs_w)
            extras_f = paths_taken[: ,:, :, obs_w*obs_h:]
        else: 
            obs_f = paths_taken.reshape(episode.episode_len, episode.n_agents, self.max_depth, obs_h, obs_w)
            extras_f = None

        return obs_f, extras_f, actions_comp, agent_pos, None, None, None # Inelegant patch to be symmetric with ind (we use those to send data if abstract)
    
    @staticmethod
    def load(filedir: str) -> "SoftDecisionTree":
        """Load an experiment from disk."""
        with open(filedir, "rb") as f:
            sdt: SoftDecisionTree = pickle.load(f)
        addition = f"{"_extra" if sdt.extras else ""}{"_abstract" if sdt.abstract else ""}"
        if sdt.agent_id is None: sdt.load_state_dict(torch.load(str(pathlib.Path(sdt.logdir,f"comp_sdt{addition}.params"))))
        else: sdt.load_state_dict(torch.load(str(pathlib.Path(sdt.logdir,f"ag{sdt.agent_id}_sdt{addition}.params"))))
        return sdt
    
    def load_best(self):
        addition = f"{"_extra" if self.extras else ""}{"_abstract" if self.abstract else ""}"
        if self.agent_id is None: self.load_state_dict(torch.load(str(pathlib.Path(self.logdir,f"comp_sdt{addition}.params"))))
        else: self.load_state_dict(torch.load(str(pathlib.Path(self.logdir,f"ag{self.agent_id}_sdt{addition}.params"))))

    def save_best(self):
        os.makedirs(self.logdir, exist_ok=True)
        addition = f"{"_extra" if self.extras else ""}{"_abstract" if self.abstract else ""}"
        if self.agent_id is None:
            torch.save(self.state_dict(), str(pathlib.Path(self.logdir,f"comp_sdt{addition}.params")))
            with open(self.logdir/f"comp_sdt_distil{addition}.pkl", 'wb') as output_file:
                pickle.dump(self, output_file)
        else:
            torch.save(self.state_dict(), str(pathlib.Path(self.logdir,f"ag{self.agent_id}_sdt{addition}.params")))
            with open(self.logdir/f"ag{self.agent_id}_sdt_distil{addition}.pkl", 'wb') as output_file:
                pickle.dump(self, output_file)