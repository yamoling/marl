from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from marlenv import Episode, Transition

from marl.models import EpisodeMemory, QNetwork, ReplayMemory, TransitionMemory
from marl.models.batch import EpisodeBatch, TransitionBatch

from .spec import HavenSpec


@dataclass
class HavenWorkerRecord:
    """A primitive transition paired with its complete macro context."""

    transition: Transition
    macro: Transition
    needs_bootstrap_goal: bool = False


@dataclass
class HavenAssemblyResult:
    """Macro interval and worker records completed by one collection step."""

    macro: Transition | None = None
    workers: tuple[HavenWorkerRecord, ...] = ()


class HavenRolloutAssembler:
    """Turn a chronological primitive stream into complete HAVEN intervals."""

    def __init__(self, spec: HavenSpec):
        self.spec = spec
        self._interval = list[Transition]()
        self._pending_boundary: HavenWorkerRecord | None = None
        self._previous_meta_action: np.ndarray | None = None
        self._steps = 0

    @property
    def steps(self) -> int:
        return self._steps

    def add(self, transition: Transition) -> HavenAssemblyResult:
        """Consume one primitive transition and emit every newly complete record. @ai-generated"""
        action = self.spec.validate_action(transition["meta_actions"])
        goal = self.spec.encode_goal(action)
        item = self._copy_worker_transition(transition)
        ready = list[HavenWorkerRecord]()
        if self._pending_boundary is not None:
            self._pending_boundary.transition.next_obs.extras[:, -self.spec.n_subgoals :] = goal
            ready.append(self._pending_boundary)
            self._pending_boundary = None
        if self._interval and not np.array_equal(action, self._interval[0]["meta_actions"]):
            raise ValueError("Macro actions must remain fixed for k primitive steps")
        self._interval.append(item)
        self._steps += 1
        if len(self._interval) < self.spec.k and not item.is_terminal:
            return HavenAssemblyResult(workers=tuple(ready))

        macro = self._make_macro_transition(action)
        records = [HavenWorkerRecord(worker, macro) for worker in self._interval]
        if item.truncated and not item.done and len(self._interval) == self.spec.k:
            records[-1].needs_bootstrap_goal = True
        elif not item.is_terminal:
            self._pending_boundary = records.pop()
        ready.extend(records)
        self._interval = []
        self._previous_meta_action = None if item.is_terminal else action
        return HavenAssemblyResult(macro=macro, workers=tuple(ready))

    def finish_episode(self):
        """Validate and reset the stream after its terminal or truncated transition. @ai-generated"""
        if self._interval or self._pending_boundary is not None:
            raise ValueError("HAVEN episode ended before its final transition was marked done or truncated")
        self._previous_meta_action = None
        self._steps = 0

    def _copy_worker_transition(self, transition: Transition) -> Transition:
        """Copy a primitive transition and inject its current subgoal. @ai-generated"""
        obs = self.spec.worker_observation(transition.obs, transition["meta_actions"])
        next_obs = self.spec.worker_observation(transition.next_obs, transition["meta_actions"])
        return Transition(
            obs=obs,
            state=transition.state,
            action=np.asarray(transition.action).copy(),
            reward=transition.reward.copy(),
            done=transition.done,
            info=transition.info,
            next_obs=next_obs,
            next_state=transition.next_state,
            truncated=transition.truncated,
            **transition.other,
        )

    def _make_macro_transition(self, action: np.ndarray) -> Transition:
        """Aggregate the active primitive interval into one macro transition. @ai-generated"""
        first, last = self._interval[0], self._interval[-1]
        return Transition(
            obs=self.spec.meta_observation(first.obs, self._previous_meta_action),
            state=first.state,
            action=action,
            reward=np.sum([transition.reward for transition in self._interval], axis=0),
            next_obs=self.spec.meta_observation(last.next_obs, action),
            next_state=last.next_state,
            done=last.done,
            truncated=last.truncated,
            info={},
        )


@dataclass
class HavenWorkerSample:
    """Worker batch with aligned macro context and representation-specific mapping."""

    workers: EpisodeBatch | TransitionBatch
    macro: EpisodeBatch | TransitionBatch
    bootstrap_goal_mask: torch.Tensor
    k: int

    def expand_macro_signal(self, signal: torch.Tensor) -> torch.Tensor:
        """Map one signal per macro interval to the corresponding worker items. @ai-generated"""
        if isinstance(self.workers, EpisodeBatch):
            signal = signal.repeat_interleave(self.k, dim=0)[: self.workers.rewards.shape[0]]
        return signal.masked_fill(self.workers.masked_indices, 0)

    def macro_bootstrap_qvalues(self, network: QNetwork) -> torch.Tensor:
        """Evaluate macro Q-values with the history required by the replay representation. @ai-generated"""
        if isinstance(self.macro, EpisodeBatch):
            return network.batch_qvalues(self.macro.all_obs, self.macro.all_extras, masks=self.macro.all_masks)
        return network.batch_qvalues(self.macro.next_obs, self.macro.next_extras)

    def apply_bootstrap_goals(self, qvalues: torch.Tensor, n_subgoals: int):
        """Fill unknown time-limit boundary goals from the current greedy macro policy. @ai-generated"""
        if not self.bootstrap_goal_mask.any():
            return
        if isinstance(self.workers, EpisodeBatch):
            for t, b in self.bootstrap_goal_mask.nonzero().tolist():
                action = qvalues[(t + 1) // self.k, b].argmax(dim=-1)
                goal = torch.nn.functional.one_hot(action, n_subgoals).to(self.workers.next_extras.dtype)
                self.workers.next_extras[t, b, :, -n_subgoals:] = goal
                self.workers.all_extras[t + 1, b, :, -n_subgoals:] = goal
            return
        action = qvalues.argmax(dim=-1)
        goal = torch.nn.functional.one_hot(action, n_subgoals).to(self.workers.next_extras.dtype)
        self.workers.next_extras[self.bootstrap_goal_mask, :, -n_subgoals:] = goal[self.bootstrap_goal_mask]


class _HavenReplayBackend:
    def __init__(self, meta_memory: ReplayMemory, worker_memory: ReplayMemory, k: int):
        self.meta_memory = meta_memory
        self.worker_memory = worker_memory
        self.k = k

    def add(self, result: HavenAssemblyResult):
        raise NotImplementedError

    def finish_episode(self):
        raise NotImplementedError

    def sample_workers(self, batch_size: int, device: torch.device) -> HavenWorkerSample:
        raise NotImplementedError


class _TransitionHavenReplay(_HavenReplayBackend):
    def __init__(self, meta_memory: TransitionMemory, worker_memory: TransitionMemory, k: int):
        super().__init__(meta_memory, worker_memory, k)
        self.contexts = deque[tuple[Transition, bool]](maxlen=worker_memory.max_size)

    def add(self, result: HavenAssemblyResult):
        """Store ready primitive records and macro transitions independently. @ai-generated"""
        if result.macro is not None:
            self.meta_memory.add_transition(result.macro)
        for record in result.workers:
            self.worker_memory.add_transition(record.transition)
            self.contexts.append((record.macro, record.needs_bootstrap_goal))

    def finish_episode(self):
        """Transition records are committed as soon as their context is complete."""

    def sample_workers(self, batch_size: int, device: torch.device) -> HavenWorkerSample:
        """Sample primitive transitions and their parallel macro snapshots. @ai-generated"""
        indices = np.random.choice(len(self.worker_memory), batch_size, replace=False)
        workers = self.worker_memory.get_batch(indices).to(device)
        contexts = [self.contexts[index] for index in indices]
        macro = TransitionBatch([context[0] for context in contexts], device=device)
        mask = torch.tensor([context[1] for context in contexts], dtype=torch.bool, device=device)
        assert isinstance(workers, TransitionBatch)
        return HavenWorkerSample(workers, macro, mask, self.k)


class _EpisodeHavenReplay(_HavenReplayBackend):
    def __init__(self, meta_memory: EpisodeMemory, worker_memory: EpisodeMemory, k: int):
        super().__init__(meta_memory, worker_memory, k)
        self.contexts = deque[Episode](maxlen=worker_memory.max_size)
        self._workers = list[Transition]()
        self._macros = list[Transition]()

    def add(self, result: HavenAssemblyResult):
        """Accumulate paired trajectories until the episode is complete. @ai-generated"""
        if result.macro is not None:
            self._macros.append(result.macro)
        self._workers.extend(record.transition for record in result.workers)

    def finish_episode(self):
        """Commit aligned worker and macro episodes to their respective stores. @ai-generated"""
        if not self._workers or not self._macros:
            raise ValueError("Cannot store an empty HAVEN episode")
        workers = Episode.from_transitions(self._workers)
        macro = Episode.from_transitions(self._macros)
        self.worker_memory.add_episode(workers)
        self.meta_memory.add_episode(macro)
        self.contexts.append(macro)
        self._workers = []
        self._macros = []

    def sample_workers(self, batch_size: int, device: torch.device) -> HavenWorkerSample:
        """Sample worker episodes with their already aligned macro episodes. @ai-generated"""
        indices = np.random.choice(len(self.worker_memory), batch_size, replace=False)
        workers = self.worker_memory.get_batch(indices).to(device)
        macro = EpisodeBatch([self.contexts[index] for index in indices], device=device)
        assert isinstance(workers, EpisodeBatch)
        mask = torch.zeros_like(workers.masks, dtype=torch.bool)
        for b, episode in enumerate(workers._base_episodes):
            if not episode.is_done and len(episode) % self.k == 0:
                mask[len(episode) - 1, b] = True
        return HavenWorkerSample(workers, macro, mask, self.k)


class HavenReplay:
    """Coordinate HAVEN interval assembly, dual storage and aligned sampling."""

    def __init__(self, spec: HavenSpec, meta_memory: ReplayMemory, worker_memory: ReplayMemory):
        """Select a replay strategy from two compatible underlying memories. @ai-generated"""
        self.spec = spec
        self.meta_memory = meta_memory
        self.worker_memory = worker_memory
        if isinstance(meta_memory, TransitionMemory) and isinstance(worker_memory, TransitionMemory):
            self.mode: Literal["transition", "episode"] = "transition"
            self._backend = _TransitionHavenReplay(meta_memory, worker_memory, spec.k)
        elif isinstance(meta_memory, EpisodeMemory) and isinstance(worker_memory, EpisodeMemory):
            self.mode = "episode"
            self._backend = _EpisodeHavenReplay(meta_memory, worker_memory, spec.k)
        else:
            raise TypeError("HAVEN requires two TransitionMemory instances or two EpisodeMemory instances")
        self._assembler = HavenRolloutAssembler(spec)

    def add_transition(self, transition: Transition):
        """Collect one chronological primitive transition into the selected backend. @ai-generated"""
        self._backend.add(self._assembler.add(transition))

    def finish_episode(self, episode: Episode):
        """Complete collection, accepting episode-only callers as a compatibility path. @ai-generated"""
        if self._assembler.steps == 0:
            actions = episode["meta_actions"]
            for t, transition in enumerate(episode.transitions()):
                transition.other["meta_actions"] = actions[t]
                self.add_transition(transition)
        elif self._assembler.steps != len(episode):
            raise ValueError("HAVEN received an episode inconsistent with its collected transitions")
        self._assembler.finish_episode()
        self._backend.finish_episode()

    def can_sample_meta(self, batch_size: int) -> bool:
        return self.meta_memory.can_sample(batch_size)

    def can_sample_workers(self, batch_size: int) -> bool:
        return self.worker_memory.can_sample(batch_size)

    def sample_meta(self, batch_size: int, device: torch.device) -> EpisodeBatch | TransitionBatch:
        """Sample ordinary macro experience for Q and V learning. @ai-generated"""
        batch = self.meta_memory.sample(batch_size).to(device)
        assert isinstance(batch, (EpisodeBatch, TransitionBatch))
        return batch

    def sample_workers(self, batch_size: int, device: torch.device) -> HavenWorkerSample:
        """Sample worker experience together with the macro context that generated it. @ai-generated"""
        return self._backend.sample_workers(batch_size, device)

    def assemble_episode(self, episode: Episode) -> tuple[Episode, Episode]:
        """Build aligned worker and macro episodes without changing replay state. @ai-generated"""
        assembler = HavenRolloutAssembler(self.spec)
        workers = list[Transition]()
        macros = list[Transition]()
        actions = episode["meta_actions"]
        for t, transition in enumerate(episode.transitions()):
            transition.other["meta_actions"] = actions[t]
            result = assembler.add(transition)
            workers.extend(record.transition for record in result.workers)
            if result.macro is not None:
                macros.append(result.macro)
        assembler.finish_episode()
        return Episode.from_transitions(workers), Episode.from_transitions(macros)
