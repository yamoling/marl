"""Regression coverage for extrinsic-only intrinsic-reward module updates (audit I4)."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from marlenv import Episode, Transition
from marlenv.catalog import DiscreteMockEnv

from marl.algos.acer import ACER
from marl.algos.dqn import DQN
from marl.algos.ppo import PPO
from marl.models import TransitionMemory
from marl.models.batch import EpisodeBatch, TransitionBatch
from marl.nn.model_bank.actor_critics import CategoricalLinearActor
from marl.nn.model_bank.qnetworks import QMLP


class RewardSpy:
    """An IR module whose update mutates rewards to expose reward-buffer aliasing."""

    def __init__(self):
        self.received = []

    def compute(self, batch):
        return torch.full_like(batch.rewards, 100.0)

    def update(self, batch, time_step):
        self.received.append(batch.rewards.clone())
        batch.rewards.add_(1000.0)
        return {"ir-updated": 1.0}


def make_transitions():
    """Collect a short two-agent rollout for the training-path checks."""
    env = DiscreteMockEnv(n_agents=2, n_actions=3, end_game=3)
    obs, state = env.reset()
    transitions = []
    for _ in range(3):
        action = env.sample_action()
        step = env.step(action)
        transitions.append(Transition.from_step(obs, state, action, step))
        obs, state = step.obs, step.state
    return env, transitions


@pytest.mark.parametrize("independent", [True, False])
def test_dqn_updates_ir_from_extrinsic_rewards_without_changing_training_rewards(monkeypatch, independent):
    """DQN's individual-learner expansion must happen before snapshotting the reward."""
    env, transitions = make_transitions()
    memory = TransitionMemory(10)
    for transition in transitions:
        memory.add(transition)
    trainer = DQN(QMLP.from_env(env, hidden_sizes=(8,)), memory=memory, batch_size=2)
    if not independent:
        trainer.mixer = cast(Any, SimpleNamespace())
    spy = RewardSpy()
    trainer.ir_module = cast(Any, spy)
    sampled = TransitionBatch(transitions[:2])
    monkeypatch.setattr(memory, "can_sample", lambda _: True)
    monkeypatch.setattr(memory, "sample", lambda _: sampled)
    seen = []
    monkeypatch.setattr(trainer, "train", lambda _, batch: seen.append(batch.rewards.clone()) or {})
    monkeypatch.setattr(trainer.policy, "update", lambda _: {})
    monkeypatch.setattr(trainer.target_updater, "update", lambda _: {})

    assert trainer._update(3)["ir-updated"] == 1.0
    torch.testing.assert_close(seen[0], spy.received[0] + 100)
    torch.testing.assert_close(sampled.rewards, seen[0])


def test_ppo_updates_ir_from_extrinsic_rewards_without_changing_training_rewards():
    """The reward-dependent IR update cannot mutate PPO's augmented rollout."""
    _, transitions = make_transitions()
    batch = TransitionBatch(transitions).for_individual_learners()
    trainer = SimpleNamespace(ir_module=RewardSpy())
    original = batch.rewards.clone()
    assert PPO.add_intrinsic_rewards(cast(PPO, trainer), batch, 3) == {"ir-updated": 1.0}
    torch.testing.assert_close(trainer.ir_module.received[0], original)
    torch.testing.assert_close(batch.rewards, original + 100)


def test_acer_updates_ir_only_on_policy_from_extrinsic_rewards(monkeypatch):
    """ACER's Retrace target keeps the bonus for both on-policy and replay updates."""
    env = DiscreteMockEnv(n_agents=2, n_actions=3, end_game=3)
    actor = CategoricalLinearActor.from_env(env, mlp_sizes=(8,))
    critic = QMLP.from_env(env, hidden_sizes=(8,))
    trainer = ACER(actor, critic, None, trust_region=False)
    spy = RewardSpy()
    trainer.ir_module = cast(Any, spy)
    agent = trainer.make_agent()
    obs, state = env.reset()
    agent.new_episode()
    episode = Episode.new(obs, state)
    for _ in range(3):
        choice = agent.choose_action(obs)
        step = env.step(choice.action)
        episode.add(Transition.from_step(obs, state, choice.action, step, **choice.details))
        obs, state = step.obs, step.state
    batch = EpisodeBatch([episode])
    extrinsic = batch.rewards.clone().unsqueeze(-1).expand(-1, -1, 2)
    seen = []
    original_retrace = trainer._retrace

    def capture_retrace(batch, *args):
        seen.append(batch.rewards.clone())
        return original_retrace(batch, *args)

    monkeypatch.setattr(trainer, "_retrace", capture_retrace)
    trainer._update(batch, 3, on_policy=True)
    trainer._update(EpisodeBatch([episode]), 3, on_policy=False)
    assert len(spy.received) == 1
    torch.testing.assert_close(spy.received[0], extrinsic)
    for rewards in seen:
        torch.testing.assert_close(rewards, extrinsic + 100)
