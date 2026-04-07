from copy import deepcopy
from dataclasses import KW_ONLY, InitVar, dataclass, field
from typing import Any

import torch
from marlenv import Transition

from marl.models import Batch, Policy, Trainer, TransitionMemory
from marl.models.batch import TransitionBatch
from marl.models.nn import Mixer
from marl.models.nn.options import OptionCritic
from marl.policy import ArgMax, EpsilonGreedy


@dataclass
class OptionCriticTrainer(Trainer):
    oc: OptionCritic
    n_agents: int
    _: KW_ONLY
    n_options: int = 4
    mixer: Mixer | None = None
    batch_size: int = 32
    memory_size: InitVar[int] = 10_000
    gamma: float = 0.99
    lr: float = 1e-4
    termination_reg: float = 0.01
    entropy_reg: float = 0.01
    option_train_policy: Policy = field(default_factory=lambda: EpsilonGreedy.constant(0.1))

    def __post_init__(self, memory_size: int):
        super().__init__()
        self.target_oc = deepcopy(self.oc)
        self.target_mixer = deepcopy(self.mixer)
        self.optim = torch.optim.Adam(self.oc.parameters(), lr=self.lr)
        self.agent = self.make_agent()
        self.memory = TransitionMemory(memory_size)

    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        transition["options"] = self.agent.options
        tb = TransitionBatch([transition], self.device)
        policy_loss, term_loss = self.actor_loss(tb)
        loss = policy_loss + term_loss
        logs = {
            "termination loss": term_loss.item(),
            "policy loss": policy_loss.item(),
            **self.option_train_policy.update(time_step),
        }
        self.memory.add(transition)
        if self.memory.can_sample(self.batch_size):
            critic_loss = self.critic_loss(self.memory.sample(self.batch_size).to(self.device))
            loss = loss + critic_loss
            logs["critic loss"] = critic_loss.item()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return logs

    def critic_loss(self, batch: Batch):
        options = batch["options"].unsqueeze(-1)
        q_options = self.oc.compute_q_options(batch.obs, batch.extras)
        q_options = torch.gather(q_options, dim=-1, index=options).squeeze(-1)
        if self.mixer is not None:
            q_options = self.mixer.forward(q_options, batch.states)
        with torch.no_grad():
            next_values = self.target_oc.value_on_arrival(batch.next_obs, batch.next_extras, options)
            if self.target_mixer is not None:
                next_values = self.target_mixer.forward(next_values, batch.next_states)
        targets = batch.rewards + self.gamma * batch.not_dones * next_values
        return torch.nn.functional.mse_loss(q_options, targets)

    def actor_loss(self, batch: Batch):
        """The actor loss is made of two parts: the policy loss and the termination loss."""
        options = batch["options"].unsqueeze(-1)
        with torch.no_grad():
            # Compute the advantage of the current options
            q_options = self.oc.compute_q_options(batch.obs, batch.extras)
            max_q_options = q_options.max(-1).values
            q_options = torch.gather(q_options, dim=-1, index=options).squeeze(-1)
            next_values = self.target_oc.value_on_arrival(batch.next_obs, batch.next_extras, options)
            if self.target_mixer is not None and self.mixer is not None:
                q_options = self.mixer.forward(q_options, batch.states)
                next_values = self.target_mixer.forward(next_values, batch.next_states)
                max_q_options = self.target_mixer.forward(max_q_options, batch.states)
        values = batch.rewards + self.gamma * batch.not_dones * next_values
        advantages = values - q_options
        # Compute the log-probability of the taken options
        dist = self.oc.policy(batch.obs, batch.extras, batch.available_actions, options.tolist())
        log_probs = dist.log_prob(batch.actions).squeeze(0)
        entropies = dist.entropy().squeeze(0)
        policy_loss = -log_probs * advantages - self.entropy_reg * entropies
        policy_loss = torch.mean(policy_loss)

        # Compute the termination loss (squeeze batch dim)
        termination_probs = self.oc.termination_probability(batch.obs, batch.extras, options).squeeze(0)
        termination_loss = termination_probs * (q_options - max_q_options) + self.termination_reg * termination_probs
        termination_loss = torch.mean(termination_loss)
        return policy_loss, termination_loss

    def make_agent(self, test_policy: Policy | None = None):
        from marl.agents import OptionAgent

        if test_policy is None:
            test_policy = ArgMax()
        return OptionAgent(self.n_options, self.n_agents, self.oc, self.option_train_policy, test_policy)
