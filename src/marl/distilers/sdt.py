import os
import time

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from typing import Optional, Any

from marl.models.batch import Batch

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
        self.lmbda = lmbda * 2 ** (-depth) # Lambda decays with depth
        self.device = device
        # Modules
        self.fc = nn.Linear(self.input_shape, 1)
        self.beta = nn.Parameter(torch.randn(1))
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
            self.left = LeafNode(self.output_shape, self.device)
            self.right = LeafNode(self.output_shape, self.device)

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
        self.path_prob = path_prob
        left_leaf_accumulator = self.left.cal_prob(x, path_prob * (1-self.prob))
        right_leaf_accumulator = self.right.cal_prob(x, path_prob * self.prob)
        self.leaf_accumulator.extend(left_leaf_accumulator)
        self.leaf_accumulator.extend(right_leaf_accumulator)
        return(self.leaf_accumulator)

    def get_penalty(self):
        penalty = (torch.sum(self.prob * self.path_prob) / torch.sum(self.path_prob), self.lmbda)
        if not self.left.leaf:
            left_penalty = self.left.get_penalty()
            right_penalty = self.right.get_penalty()
            self.penalties.append(penalty)
            self.penalties.extend(left_penalty)
            self.penalties.extend(right_penalty)
        return(self.penalties)


class LeafNode(nn.Module):
    """SoftDecisionTree: a class representing a Leaf Node in a SDT
    Almost copy pasted from: https://github.com/kimhc6028/soft-decision-tree/"""
    def __init__(self,
                 output_shape: tuple[int, ...],
                 device: Optional[torch.device] = None
                 ):
        super(LeafNode, self).__init__()
        self.output_shape = output_shape
        self.device = device

        self.param = nn.Parameter(torch.randn(self.output_shape))
        self.softmax = nn.Softmax(dim=1)

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
    logdir: str
    output_shape: int
    max_depth: int
    lr: float
    lmbda: float
    momentum: float
    device: Optional[torch.device]
    seed: int
    log_interval: int # not sure I need, enforced/done by DQN?

    def __init__(self, 
                 input_shape: int,
                 output_shape: int,
                 logdir: str,
                 max_depth: int = 4, 
                 seed: int = 0,
                 log_interval: int = 50, # not sure I need
                 bs: int = 64,
                 lr: Optional[float] = 0.001,
                 lmbda: Optional[float] = 0.01,
                 momentum: Optional[float] = 0.01,
                 device: Optional[torch.device] = torch.device('cpu'),
                 ):
        super(SoftDecisionTree, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.logdir = logdir

        self.max_depth = max_depth
        self.seed = seed
        self.log_interval = log_interval

        self.batch_size = bs
        self.lr = lr
        self.lmbda = lmbda
        self.momentum = momentum
        self.device = device

        self.root = InnerNode(1, self.input_shape, self.output_shape, self.max_depth, self.lmbda, self.device)

        self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)

        self.test_acc = []
        self.define_extras(self.batch_size)
        self.best_accuracy = 0.0
    
    def check(self):
        pass

    def define_extras(self, batch_size):
        self.path_prob_init = torch.ones(batch_size, 1, device=self.device)
     
    def cal_loss(self, x, y):
        batch_size = y.shape[0]
        leaf_accumulator = self.root.cal_prob(x, self.path_prob_init)
        loss = 0.
        max_prob = [-1. for _ in range(batch_size)]
        max_Q = torch.zeros((batch_size,self.output_shape))
        for (path_prob, Q) in leaf_accumulator:
            TQ = torch.bmm(y.view(batch_size, 1, self.output_shape), torch.log(Q).view(batch_size, self.output_shape, 1)).view(-1,1)
            loss += path_prob * TQ
            path_prob_numpy = path_prob.cpu().data.numpy().reshape(-1)
            for i in range(batch_size):
                if max_prob[i] < path_prob_numpy[i]:
                    max_prob[i] = path_prob_numpy[i]
                    max_Q[i] = Q[i]
        loss = loss.mean()
        penalties = self.root.get_penalty()
        C = 0.
        for (penalty, lmbda) in penalties:
            C -= lmbda * 0.5 *(torch.log(penalty) + torch.log(1-penalty))
        #output = torch.stack(max_Q)
        self.root.reset() ##reset all stacked calculation
        return(-loss + C, max_Q) ## -log(loss) will always output non, because loss is always below zero. I suspect this is the mistake of the paper?


    def train_(self, train_data, train_targets, epoch=0):
        """
        While the model outputs a distribution over actions, train_data should have the one-hot encoded action choice"""
        self.train()
        torch.manual_seed(self.seed+epoch)
        self.define_extras(self.batch_size)
        train_acc = []
        for batch_idx  in range (len(train_data)):
            correct = 0
            target = torch.Tensor(train_targets[batch_idx], device=self.device)
            data = torch.Tensor(train_data[batch_idx], device=self.device)

            batch_size = target.shape[0]

            if not batch_size == self.batch_size: #because we have to initialize parameters for batch_size, tensor not matches with batch size cannot be trained
                self.define_extras(batch_size)

            self.optimizer.zero_grad()

            loss, output = self.cal_loss(data, target)
            #loss.backward(retain_variables=True)
            loss.backward()
            self.optimizer.step()
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.max(1)[1]).cpu().sum()
            accuracy = 100. * correct / len(data)

            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.4f}%)'.format(
                    epoch, batch_idx, len(train_data),
                    100. * batch_idx / len(train_data), loss.item(),
                    correct, len(data),
                    accuracy))
                train_acc.append(accuracy)
        print(f"Train accuracy for epoch {epoch}: {np.mean(train_acc)}\n ")

    def test_(self, train_data, train_targets, epoch=0):
        self.eval()
        
        self.define_extras(self.batch_size)
        test_loss = 0
        correct = 0
        for batch_idx  in range (len(train_data)):

            target = torch.Tensor(train_targets[batch_idx], device=self.device)
            data = torch.Tensor(train_data[batch_idx], device=self.device)

            batch_size = target.shape[0]

            if not batch_size == self.batch_size: #because we have to initialize parameters for batch_size, tensor not matches with batch size cannot be trained
                self.define_extras(batch_size)

            _, output = self.cal_loss(data, target)
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.max(1)[1]).cpu().sum()
        accuracy = 100.*correct/(len(train_data)*len(train_data[0]))
        print('\nTest set: Accuracy: {}/{} ({:.4f}%)\n'.format(
            correct, len(train_data)*len(train_data[0]),
            accuracy))
        self.test_acc.append(accuracy)

        if accuracy > self.best_accuracy:
            self.save_best()
            self.best_accuracy = accuracy

    def save_best(self):
        with open(os.path.join(self.logdir, 'sdt_distil.pkl'), 'wb') as output_file:
            pickle.dump(self, output_file)