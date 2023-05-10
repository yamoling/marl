import os
import torch
import numpy as np
from rlenv import Observation, Episode, Transition
from marl.models import RLAlgo, TransitionMemory
from marl import nn
from marl.utils import get_device



class PPO(RLAlgo):
    def __init__(
        self,
        lr_actor: float, 
        lr_critic: float, 
        gamma: float, 
        ac_network: nn.ActorCriticNN,
        clip_eps: float = 0.2,
        c1: float = 0.5, # C1 and C2 from the paper equation 9
        c2: float = 0.01,
        batch_size=1000
    ):
        self.gamma = gamma
        self.device = get_device()
        self.network = ac_network.to(self.device)
        self.optimizer = torch.optim.Adam([
            {"params": self.network.value_parameters, "lr": lr_critic},
            {"params": self.network.policy_parameters, "lr": lr_actor}
        ])
        self.is_training = True
        self.action_probs: np.ndarray = []
        self.clip_low = 1 - clip_eps
        self.clip_high = 1 + clip_eps
        self.eps = 0.2
        self.c1 = c1
        """Factor of the critic loss from the paper equation 9."""
        self.c2 = c2
        """Factor of the entropy loss from the paper equation 9."""
        self.memory = TransitionMemory(batch_size)

    def choose_action(self, observation: Observation) -> np.ndarray[np.int64]:
        with torch.no_grad():
            obs_data = torch.tensor(observation.data).to(self.device, non_blocking=True)
            logits = self.network.policy(obs_data)
            logits[torch.tensor(observation.available_actions) == 0] = -torch.inf
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            if self.is_training:
                probs = torch.gather(dist.probs, dim=-1, index=action.unsqueeze(-1))
                self.action_probs = probs.numpy(force=True)
            return action.numpy(force=True)
        
    def before_test_episode(self, time_step: int, test_num: int):
        self.is_training = False
    
    def after_tests(self, episodes: list[Episode], time_step: int):
        self.is_training = True

    def after_train_step(self, transition: Transition, time_step: int):
        transition.action_probs = self.action_probs
        self.memory.add(transition)
        if len(self.memory) == self.memory.max_size:
            self.learn()
        return super().after_train_step(transition, time_step)

    def learn(self):
        batch = self.memory.get_batch(range(len(self.memory))).to(self.device)
        self.memory.clear()
        batch.actions = batch.actions.squeeze(-1)
        
        with torch.no_grad():
            last_obs_next_value = None
            if batch.dones[-1] != 1:
                last_obs_next_value = self.network.value(batch.obs_[-1])
                last_obs_next_value = last_obs_next_value.squeeze(-1)
            returns = batch.compute_normalized_returns(self.gamma, last_obs_value=last_obs_next_value)
            old_logits, predicted_values = self.network.forward(batch.obs)
            predicted_values = predicted_values.squeeze(-1) # Squeeze last dimension of shape [1]
            advantage = returns.unsqueeze(-1) - predicted_values
            old_log_probs = torch.distributions.Categorical(logits=old_logits).log_prob(batch.actions)

        for _ in range(20):
            logits, values = self.network.forward(batch.obs)
            values = values.squeeze(-1)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(batch.actions)

            # Compute ratio between new and old probabilities in the log space (basically importance sampling)
            rho = torch.exp(log_probs - old_log_probs)

            # Actor surrogate loss
            surrogate_1 = rho * advantage
            surrogate_2 = torch.clip(rho, min=self.clip_low, max=self.clip_high) * advantage
            actor_loss = torch.min(surrogate_1, surrogate_2).mean()
            # Value estimation loss
            critic_loss = torch.mean((values - returns) ** 2)
            # Entropy loss
            entropy_loss: torch.Tensor = torch.mean(dist.entropy())

            self.optimizer.zero_grad()
            # Maximize actor loss, minimize critic loss and maximize entropy loss
            loss = -actor_loss + self.c1 * critic_loss - self.c2 * entropy_loss
            loss.backward()
            self.optimizer.step()

    def summary(self) -> dict:
        return {
            **super().summary(),
            "gamma": self.gamma,
            "lr_critic": self.optimizer.param_groups[0]["lr"],
            "lr_actor": self.optimizer.param_groups[1]["lr"],
            "clip_eps": self.clip_high - 1,
            "c1": self.c1,
            "c2": self.c2,
            "ac_network": self.network.summary()
        }

    @classmethod
    def from_summary(cls, summary: dict) -> "PPO":
        ac_network = nn.ActorCriticNN.from_summary(summary.pop("ac_network"))
        return PPO(ac_network=ac_network, **summary)

    def save(self, to_path: str):
        os.makedirs(to_path, exist_ok=True)
        file_path = os.path.join(to_path, 'ac_network')
        torch.save(self.network.state_dict(), file_path)
    
    def load(self, from_path: str):
        file_path = os.path.join(from_path, 'ac_network')
        self.network.load_state_dict(torch.load(file_path))    


        