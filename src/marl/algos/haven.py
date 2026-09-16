from copy import copy, deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from marlenv import Episode, Observation, Transition

from marl.agents import Haven
from marl.models import EpisodeMemory, Mixer, QNetwork, Trainer
from marl.models.batch import EpisodeBatch

from .dqn import DQN


@dataclass
class HavenTrainer(Trainer):
    """HAVEN with episode replay and independently optimized macro Q, V and worker Q.

    The two DQN trainers supply replay, optimizers, discounts and update schedules.
    A one-output QNetwork represents each agent's scalar V; by default it is a
    fresh instance of the macro architecture. Its mixer is independent too.
    See doc/haven.md for equations, input layout and reference differences.
    """

    meta_trainer: DQN
    worker_trainer: DQN
    n_workers: int
    n_subgoals: int
    k: int
    n_meta_extras: int
    n_agent_extras: int
    n_meta_warmup_steps: int = 0
    value_network: QNetwork | None = None
    value_mixer: Mixer | None = None
    value_lr: float | None = None
    use_previous_meta_action: bool = True

    def __post_init__(self):
        """Validate the hierarchy and construct the independent value estimator. @ai-generated"""
        super().__post_init__()
        if min(self.k, self.n_workers, self.n_subgoals) <= 0:
            raise ValueError("k, n_workers and n_subgoals must be positive")
        if min(self.n_meta_extras, self.n_agent_extras, self.n_meta_warmup_steps) < 0:
            raise ValueError("Extras sizes and warmup must be nonnegative")
        for trainer in (self.meta_trainer, self.worker_trainer):
            if not isinstance(trainer, DQN) or not isinstance(trainer.memory, EpisodeMemory):
                raise TypeError("HAVEN requires two DQN trainers with EpisodeMemory for aligned recurrent replay")
            if trainer.mixer is None or trainer.qnetwork.n_objectives != 1:
                raise ValueError("HAVEN requires a cooperative, single-objective mixer at both levels")
            if trainer.ir_module is not None or trainer.vbe is not None:
                raise ValueError("HAVEN supplies its own intrinsic reward; additional IR/VBE is unsupported")
            if trainer.qnetwork.n_agents != self.n_workers:
                raise ValueError("Network agent counts must match n_workers")
        meta = self.meta_trainer.qnetwork
        worker = self.worker_trainer.qnetwork
        meta_extras = self.n_meta_extras + (self.n_subgoals if self.use_previous_meta_action else 0)
        if meta.n_actions != self.n_subgoals or tuple(meta.extras_shape) != (meta_extras,):
            raise ValueError("Macro network extras must include meta extras and, when enabled, the previous macro action")
        if tuple(worker.extras_shape) != (self.n_meta_extras + self.n_agent_extras + self.n_subgoals,):
            raise ValueError("Worker extras must contain meta extras, agent extras and subgoal padding")
        if self.meta_trainer.memory is self.worker_trainer.memory:
            raise ValueError("Macro and worker replay memories must be separate")
        if self.value_network is None:
            self.value_network = replace(meta, n_actions=1, duelling=False)
        if self.value_mixer is None:
            self.value_mixer = deepcopy(self.meta_trainer.mixer)
        assert self.value_mixer is not None and self.value_network is not None
        if (
            self.value_network.n_actions != 1
            or self.value_network.n_objectives != 1
            or tuple(self.value_network.obs_shape) != tuple(meta.obs_shape)
            or tuple(self.value_network.extras_shape) != tuple(meta.extras_shape)
            or self.value_network.n_agents != self.n_workers
        ):
            raise ValueError("Value network must have macro inputs and one scalar output per agent")
        groups = [list(t.target_updater.parameters) for t in (self.meta_trainer, self.worker_trainer)]
        groups.append(list(self.value_network.parameters()) + list(self.value_mixer.parameters()))
        if len({id(p) for group in groups for p in group}) != sum(map(len, groups)):
            raise ValueError("Macro Q, worker Q and value estimators must not share parameters")
        self._value_parameters = groups[-1]
        lr = self.meta_trainer.lr if self.value_lr is None else self.value_lr
        if self.meta_trainer.optimiser_type == "rmsprop":
            self.value_optimiser = torch.optim.RMSprop(self._value_parameters, lr=lr, eps=1e-5)
        else:
            self.value_optimiser = torch.optim.Adam(self._value_parameters, lr=lr)

    def _meta_observation(self, observation: Observation, previous_action=None) -> Observation:
        """Copy macro inputs without retaining worker subgoals or action masks. @ai-generated"""
        result = copy(observation)
        result.extras = observation.extras[:, : self.n_meta_extras].copy()
        if self.use_previous_meta_action:
            previous = np.zeros((self.n_workers, self.n_subgoals), dtype=np.float32)
            if previous_action is not None:
                previous = np.eye(self.n_subgoals, dtype=np.float32)[np.asarray(previous_action)]
            result.extras = np.concatenate((result.extras, previous), axis=-1)
        result.available_actions = np.ones((self.n_workers, self.n_subgoals), dtype=bool)
        return result

    def _build_meta_episode(self, episode: Episode) -> Episode:
        """Aggregate complete intervals, including every reward and the final partial interval. @ai-edited"""
        transitions = list(episode.transitions())
        if not transitions:
            raise ValueError("Cannot build an empty macro episode")
        actions = episode["meta_actions"]
        result = Episode.new(self._meta_observation(transitions[0].obs), transitions[0].state)
        for start in range(0, len(transitions), self.k):
            interval = transitions[start : start + self.k]
            first, last = interval[0], interval[-1]
            result.add(
                Transition(
                    obs=self._meta_observation(first.obs, actions[start - 1] if start else None),
                    state=first.state,
                    action=np.asarray(actions[start]).copy(),
                    reward=np.sum([t.reward for t in interval], axis=0),
                    next_obs=self._meta_observation(last.next_obs, actions[start]),
                    next_state=last.next_state,
                    done=last.done,
                    truncated=last.truncated,
                    info={},
                )
            )
        return result

    def _worker_episode(self, episode: Episode) -> Episode:
        """Reconstruct recorded subgoals without mutating environment observations. @ai-generated"""
        result = replace(episode)
        result.all_extras = [extras.copy() for extras in episode.all_extras]
        actions = episode["meta_actions"]
        for t, action in enumerate(actions):
            action = np.asarray(action)
            if action.shape != (self.n_workers,) or not np.issubdtype(action.dtype, np.integer):
                raise ValueError("HAVEN macro actions must be one discrete index per worker")
            if np.any((action < 0) | (action >= self.n_subgoals)):
                raise ValueError("Macro action outside the subgoal space")
            if t % self.k and not np.array_equal(action, actions[t - 1]):
                raise ValueError("Macro actions must remain fixed for k primitive steps")
            result.all_extras[t][:, -self.n_subgoals :] = np.eye(self.n_subgoals, dtype=np.float32)[action]
        # Inside an unfinished interval, the bootstrap still uses its current goal.
        result.all_extras[-1][:, -self.n_subgoals :] = result.all_extras[-2][:, -self.n_subgoals :]
        return result

    def update_step(self, transition: Transition, time_step: int):
        """Train from completed episodes on each child's primitive-step schedule. @ai-edited"""
        return self._update(time_step, on_episode=None)

    def update_episode(self, episode: Episode, episode_num: int, time_step: int):
        """Store aligned raw trajectories once; warmup gates learning, not collection. @ai-edited"""
        worker_episode = self._worker_episode(episode)
        self.worker_trainer.memory.add_episode(worker_episode)
        self.meta_trainer.memory.add_episode(self._build_meta_episode(worker_episode))
        return self._update(time_step, on_episode=episode_num)

    def _values(self, batch: EpisodeBatch):
        """Evaluate V along complete macro histories, including bootstrap states. @ai-generated"""
        assert self.value_network is not None and self.value_mixer is not None
        local = self.value_network.batch_qvalues(batch.all_obs, batch.all_extras, masks=batch.all_masks).squeeze(-1)
        states = torch.cat((batch.states[:1], batch.next_states))
        extras = torch.cat((batch.states_extras[:1], batch.next_states_extras))
        return self.value_mixer.forward(local, states, extras)

    def _value_targets(self, batch: EpisodeBatch):
        """Equation 8 uses the online macro Q maximum, not target Q or next V. @ai-generated"""
        trainer = self.meta_trainer
        assert trainer.mixer is not None
        with torch.no_grad():
            q = trainer.qnetwork.batch_qvalues(batch.all_obs, batch.all_extras, masks=batch.all_masks)[1:]
            values, actions = q.max(dim=-1)
            mixed = trainer.mixer.forward(
                values, batch.next_states, batch.next_states_extras, **trainer.get_mixing_kwargs(batch, q, is_next=True, actions=actions)
            )
            return batch.rewards + trainer.gamma * mixed.masked_fill(batch.dones | batch.masked_indices, 0)

    def _train_value(self, batch: EpisodeBatch):
        """Optimize only the high-level value network and its own mixer. @ai-generated"""
        values = self._values(batch)[:-1]
        loss = ((values - self._value_targets(batch)).square() * batch.masks).sum() / batch.n_items
        self.value_optimiser.zero_grad()
        loss.backward()
        clip = self.meta_trainer.grad_norm_clipping
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(self._value_parameters, clip)
        self.value_optimiser.step()
        return {"value-loss": loss.item()}

    def _intrinsic_rewards(self, worker: EpisodeBatch, meta: EpisodeBatch):
        """Recompute the detached macro advantage and divide it evenly over k steps. @ai-generated"""
        with torch.no_grad():
            values = self._values(meta)
            advantage = meta.rewards + self.meta_trainer.gamma * values[1:].masked_fill(meta.dones, 0) - values[:-1]
            reward = advantage.repeat_interleave(self.k, dim=0)[: worker.rewards.shape[0]] / self.k
            return reward.masked_fill(worker.masked_indices, 0)

    def _worker_batch(self, batch: EpisodeBatch):
        """Align macro replay with worker samples and repair time-limit boundary goals. @ai-generated"""
        episodes = batch._base_episodes
        meta = EpisodeBatch([self._build_meta_episode(e) for e in episodes], device=self.device)
        # No behavior action exists beyond a time limit. At a macro boundary use
        # the current greedy macro policy for that bootstrap, preserving history.
        with torch.no_grad():
            q = self.meta_trainer.qnetwork.batch_qvalues(meta.all_obs, meta.all_extras, masks=meta.all_masks)
        for b, episode in enumerate(episodes):
            t = len(episode)
            if not episode.is_done and t % self.k == 0:
                action = q[t // self.k, b].argmax(dim=-1)
                goal = torch.nn.functional.one_hot(action, self.n_subgoals).to(batch.all_extras.dtype)
                batch.all_extras[t, b, :, -self.n_subgoals :] = goal
                batch.next_extras[t - 1, b, :, -self.n_subgoals :] = goal
        intrinsic = self._intrinsic_rewards(batch, meta)
        batch.rewards = batch.rewards + intrinsic
        return batch, {"intrinsic-reward": (intrinsic.sum() / batch.n_items).item()}

    def _update(self, time_step: int, on_episode: int | None):
        """Respect independent child schedules while refreshing rewards at sampling time. @ai-generated"""
        logs = {}
        if time_step < self.n_meta_warmup_steps:
            return logs
        for prefix, trainer in (("meta", self.meta_trainer), ("worker", self.worker_trainer)):
            due = trainer.should_update_at(time_step=time_step) if on_episode is None else trainer.should_update_at(episode_num=on_episode)
            if not due or not trainer.memory.can_sample(trainer.batch_size):
                continue
            batch = trainer.memory.sample(trainer.batch_size).to(self.device)
            assert isinstance(batch, EpisodeBatch)
            if prefix == "meta":
                logs.update(self._train_value(batch))
            else:
                batch, reward_logs = self._worker_batch(batch)
                logs.update(reward_logs)
            child_logs = trainer.train(time_step, batch)
            child_logs.update(trainer.policy.update(time_step))
            child_logs.update(trainer.target_updater.update(time_step))
            logs.update({f"{prefix}-{key}": value for key, value in child_logs.items()})
        return logs

    def make_agent(self):
        return Haven(
            self.meta_trainer.make_agent(),
            self.worker_trainer.make_agent(),
            self.n_subgoals,
            self.n_workers,
            self.k,
            self.n_meta_extras,
            self.n_agent_extras,
            use_previous_meta_action=self.use_previous_meta_action,
        )

    def networks(self, modules: bool = False) -> list:
        return self.meta_trainer.networks(modules) + self.worker_trainer.networks(modules) + [self.value_network, self.value_mixer]

    def randomize(self, method: Literal["xavier", "orthogonal"] = "xavier"):
        """Initialize all three estimators and synchronize the two Q targets. @ai-edited"""
        self.meta_trainer.randomize(method)
        self.worker_trainer.randomize(method)
        assert self.value_network is not None and self.value_mixer is not None
        self.value_network.randomize(method)
        self.value_mixer.randomize(method)

    def to(self, device: torch.device):
        """Move all networks and existing optimizer state together. @ai-edited"""
        super().to(device)
        self.meta_trainer.to(device)
        self.worker_trainer.to(device)
        for optimizer in (self.value_optimiser, self.meta_trainer.optimiser, self.worker_trainer.optimiser):
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)
        return self

    def save(self, directory: Path):
        """Keep the three estimators in separate namespaces, including optimizer state. @ai-generated"""
        for name, trainer in (("meta", self.meta_trainer), ("worker", self.worker_trainer)):
            trainer.save(directory / name)
            torch.save(trainer.optimiser.state_dict(), directory / name / "optimizer.pt")
        value_dir = directory / "value"
        value_dir.mkdir(parents=True, exist_ok=True)
        assert self.value_network is not None and self.value_mixer is not None
        self.value_network.save(value_dir)
        self.value_mixer.save(value_dir)
        torch.save(self.value_optimiser.state_dict(), value_dir / "optimizer.pt")

    def load(self, directory: Path):
        """Restore all estimators and optimizers from their separate namespaces. @ai-generated"""
        for name, trainer in (("meta", self.meta_trainer), ("worker", self.worker_trainer)):
            trainer.load(directory / name)
            trainer.optimiser.load_state_dict(torch.load(directory / name / "optimizer.pt", weights_only=True, map_location=self.device))
        assert self.value_network is not None and self.value_mixer is not None
        self.value_network.load(directory / "value")
        self.value_mixer.load(directory / "value")
        self.value_optimiser.load_state_dict(torch.load(directory / "value" / "optimizer.pt", weights_only=True, map_location=self.device))
