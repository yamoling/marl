from copy import deepcopy
from dataclasses import KW_ONLY, InitVar, dataclass, field
from typing import Any

import torch
from marlenv import Transition

from marl.models import Batch, Policy, Trainer, TransitionMemory
from marl.models.batch import TransitionBatch
from marl.models.nn import Mixer
from marl.models.nn.options import OptionCriticNetwork
from marl.policy import ArgMax, EpsilonGreedy
from marl.training.qtarget_updater import HardUpdate, TargetParametersUpdater


@dataclass
class OptionCritic(Trainer):
    oc: OptionCriticNetwork
    n_agents: int
    _: KW_ONLY
    mixer: Mixer | None = None
    batch_size: int = 32
    critic_train_interval: int = 4
    memory_size: InitVar[int] = 10_000
    gamma: float = 0.99
    lr: float = 1e-4
    termination_reg: float = 0.01
    entropy_reg: float = 0.01
    q_updater: InitVar[TargetParametersUpdater | None] = None
    target_updater: TargetParametersUpdater = field(init=False)
    option_train_policy: Policy = field(default_factory=lambda: EpsilonGreedy.constant(0.1))

    def __post_init__(self, memory_size: int, q_updater: TargetParametersUpdater | None):
        super().__init__()
        self.target_oc = deepcopy(self.oc)
        self.target_mixer = deepcopy(self.mixer)
        self.optim = torch.optim.Adam(self.oc.parameters(), lr=self.lr)
        self.memory = TransitionMemory(memory_size)
        if q_updater is None:
            q_updater = HardUpdate(200)
        self.target_updater = q_updater
        self.target_updater.add_parameters(self.oc.parameters(), self.target_oc.parameters())
        if self.mixer is not None and self.target_mixer is not None:
            self.target_updater.add_parameters(self.mixer.parameters(), self.target_mixer.parameters())

    @property
    def n_options(self):
        return self.oc.n_options

    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        tb = TransitionBatch([transition], self.device)
        policy_loss, term_loss = self.actor_loss(tb)
        loss = policy_loss + term_loss
        logs = {
            "termination loss": term_loss.item(),
            "policy loss": policy_loss.item(),
            **self.option_train_policy.update(time_step),
        }
        self.memory.add(transition)
        if self.memory.can_sample(self.batch_size) and time_step % self.critic_train_interval == 0:
            critic_loss = self.critic_loss(self.memory.sample(self.batch_size).to(self.device))
            loss = loss + critic_loss
            logs["critic loss"] = critic_loss.item()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        logs = logs | self.target_updater.update(time_step)
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
        """The actor loss is made of two parts: the policy loss and the termination loss.

        According to Theorem 1 and 2 in Bacon et al. (2016):
        - Policy gradient uses Q_U(s, ω, a) as advantage estimate
        - Termination gradient uses advantage at next state: A_Ω(s', ω) = Q_Ω(s', ω) - V_Ω(s')
        """
        options = batch["options"].unsqueeze(-1)
        with torch.no_grad():
            # Compute current state Q-values for policy loss
            q_options = self.oc.compute_q_options(batch.obs, batch.extras)
            q_options = torch.gather(q_options, dim=-1, index=options).squeeze(-1)

            # Compute next state Q-values for termination loss (Theorem 2 - next state advantage)
            next_q_options = self.target_oc.compute_q_options(batch.next_obs, batch.next_extras)
            next_q_max = next_q_options.max(dim=-1).values
            next_q_options_continued = torch.gather(next_q_options, dim=-1, index=options).squeeze(-1)

            # Compute value at arrival for policy loss
            next_values = self.target_oc.value_on_arrival(batch.next_obs, batch.next_extras, options)

            # Apply mixer if present
            if self.target_mixer is not None and self.mixer is not None:
                q_options = self.mixer.forward(q_options, batch.states)
                next_q_options_continued = self.target_mixer.forward(next_q_options_continued, batch.next_states)
                next_q_max = self.target_mixer.forward(next_q_max, batch.next_states)
                next_values = self.target_mixer.forward(next_values, batch.next_states)

        # Policy loss: policy gradient using Q_U bootstrapped target
        values = batch.rewards + self.gamma * batch.not_dones * next_values
        policy_advantages = values - q_options
        dist = self.oc.policy(batch.obs, batch.extras, batch.available_actions, options.tolist())
        log_probs = dist.log_prob(batch.actions).squeeze(0)
        entropies = dist.entropy().squeeze(0)
        policy_loss = -log_probs * policy_advantages - self.entropy_reg * entropies
        policy_loss = torch.mean(policy_loss)

        # Termination loss (Theorem 2): use advantage at next state, mask out episode ends
        # A_Ω(s', ω) = Q_Ω(s', ω) - V_Ω(s')
        next_termination_probs = self.oc.termination_probability(batch.next_obs, batch.next_extras, options).squeeze(0)
        next_advantage = next_q_options_continued - next_q_max
        termination_loss = next_termination_probs * (next_advantage + self.termination_reg) * batch.not_dones
        termination_loss = torch.mean(termination_loss)

        return policy_loss, termination_loss

    def make_agent(self, test_policy: Policy | None = None):
        from marl.agents import OptionAgent

        if test_policy is None:
            test_policy = ArgMax()
        return OptionAgent(self.n_options, self.n_agents, self.oc, self.option_train_policy, test_policy)
