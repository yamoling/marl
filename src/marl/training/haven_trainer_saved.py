from typing import Any, Literal

from marlenv import Transition
from marl.models.trainer import Trainer
from marl.models.nn import ContinuousActorCriticNN
from marl.models import TransitionMemory
from marl.agents import Haven
import torch
from pprint import pprint


class HavenTrainer(Trainer):
    def __init__(
        self,
        worker_trainer: Trainer,
        actor_critic: ContinuousActorCriticNN,
        gamma: float,
        n_epochs: int,
        lr: float,
        eps_clip: float,
        k: int,
        c1: float,
        exploration_c2: float,
    ):
        """
        Args:
            worker_trainer (Trainer): The worker trainer to train the worker agent.
            actor_critic (ContinuousActorCriticNN): The actor critic network.
            gamma (float): The discount factor.
            n_epochs (int): The number of epochs to train the agent (PPO).
            lr (float): The learning rate.
            eps_clip (float): The clipping parameter for the PPO loss.
            k (int): The number of time steps a subgoal is maintained.
        """
        if worker_trainer.update_on_steps:
            upd = "step"
            interval = worker_trainer.step_update_interval
        else:
            upd = "episode"
            interval = worker_trainer.episode_update_interval
        super().__init__(upd, interval)
        self.worker_trainer = worker_trainer
        self.memory = TransitionMemory(20)
        self.batch_size = 20
        self.actor_critic = actor_critic
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.eps_clip = eps_clip
        self.optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)
        assert len(actor_critic.action_output_shape) == 2, "The actor should output a mean and an std"
        self.n_agents, self.n_subgoals = actor_critic.action_output_shape
        self.k = k
        self.c1 = c1
        self.c2 = exploration_c2

    def ppo_update(self, time_step: int) -> dict[str, float]:
        if not self.memory.can_sample(self.batch_size):
            return {}
        # We retrieve the whole memory sequentially (the order matters for GAE)
        batch = self.memory.get_batch(range(self.batch_size))
        batch.obs = batch.obs[:, 0]
        batch.extras = batch.extras[:, 0]
        batch.next_obs = batch.next_obs[:, 0]
        batch.next_extras = batch.next_extras[:, 0]
        batch = batch.to(self.device)
        # Add reward normalization ?
        # Add GAE ?

        meta_actions = batch["meta_actions"]
        with torch.no_grad():
            (means, stds), values = self.actor_critic.forward(batch.obs, batch.extras)
            stats = {
                "means-mean": means.mean().item(),
                "means-min": means.min().item(),
                "means-max": means.max().item(),
                "stds-mean": stds.mean().item(),
                "stds-min": stds.min().item(),
                "stds-max": stds.max().item(),
            }
            print(stats)
            policy = torch.distributions.Normal(means, stds)
            log_probs = policy.log_prob(meta_actions)

            next_values = self.actor_critic.value(batch.next_obs, batch.next_extras) * batch.dones
            # The advantage A(s, a) = Q(s, a) - V(s)
            # <=> A(s, a) = r(s, a, s') + gamma * V(s') - V(s)
            advantages = batch.compute_gae(values, self.gamma, 0.95)
            # advantages = batch.rewards + self.gamma * next_values - values
            # advantages = advantages.unsqueeze(-1).expand_as(log_probs)

        total_loss = 0.0
        min_loss = 0.0
        max_loss = 0.0
        for _ in range(self.n_epochs):
            (means, stds), values = self.actor_critic.forward(batch.obs, batch.extras)
            policy = torch.distributions.Normal(means, stds)
            new_log_probs = policy.log_prob(meta_actions)

            ratio = torch.exp(new_log_probs - log_probs)
            surrogate_actor_loss1 = ratio * advantages
            surrogate_actor_loss2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            # L^CLIP(θ) = E[ min(r(θ)A, clip(r(θ), 1 − ε, 1 + ε)A) ] in PPO paper
            actor_loss = torch.min(surrogate_actor_loss1, surrogate_actor_loss2).mean()

            next_values = self.actor_critic.value(batch.next_obs, batch.next_extras)
            target_values = batch.rewards + self.gamma * next_values
            # L^VF(θ) = E[(V(s) - V_targ(s))^2] in PPO paper
            critic_loss = (values - target_values).pow(2).mean()

            # S[\pi_0](s_t) in the paper
            entropy_loss = policy.entropy().mean()

            self.optimizer.zero_grad()
            # Equation (9) in the paper, but we minimize (instead of maximize in the paper)
            loss = -(actor_loss - self.c1 * critic_loss + self.c2 * entropy_loss)
            # loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            min_loss = min(min_loss, loss.item())
            max_loss = max(max_loss, loss.item())

        self.memory.clear()
        logs = {
            "meta-avg_loss": total_loss / self.n_epochs,
            "meta-total_loss": total_loss,
            "meta-min_loss": min_loss,
            "meta-max_loss": max_loss,
        }
        pprint(logs)
        return logs

    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        self.memory.add(transition)
        logs = self.ppo_update(time_step)
        if len(logs) > 0:
            print(f"t={time_step}")
        for key, value in self.worker_trainer.update_step(transition, time_step).items():
            logs[f"worker-{key}"] = value
        return logs

    def make_agent(self):
        return Haven(
            self.actor_critic,
            self.worker_trainer.make_agent(),
            self.n_subgoals,
            self.k,
        )

    def randomize(self, method: Literal["xavier", "orthogonal"] = "xavier"):
        self.actor_critic.randomize(method)
        self.worker_trainer.randomize(method)
