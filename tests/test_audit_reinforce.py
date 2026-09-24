"""Regressions for REINFORCE episode targets and actor/critic training."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from marlenv import Episode, Transition
from marlenv.catalog import DiscreteMockEnv

from marl.algos.reinforce import Reinforce
from marl.models.batch import EpisodeBatch
from marl.nn.model_bank import actor_critics


def make_episode(*, terminated: bool) -> tuple[DiscreteMockEnv, Episode]:
    """Build a two-step trajectory with a nontrivial legal-action mask. @ai-generated"""
    env = DiscreteMockEnv(n_agents=2, n_actions=3, end_game=2 if terminated else 10)
    obs, state = env.reset()
    transitions = []
    for _ in range(2):
        obs.available_actions[:] = [[False, True, True], [True, False, True]]
        action = np.array([1, 0])
        step = env.step(action)
        transitions.append(Transition.from_step(obs, state, action, step))
        obs, state = step.obs, step.state
    episode = Episode.from_transitions(transitions)
    if not terminated:
        episode.is_truncated = True
    return env, episode


def make_trainer(env, method="monte_carlo"):
    """Use the same actor/critic factory as PPO. @ai-generated"""
    actor, critic = actor_critics.from_env(
        env, recurrent=False, independent=True, actor_kwargs={"mlp_sizes": (8,)}, critic_kwargs={"mlp_sizes": (8,)}
    )
    return Reinforce(env.n_agents, actor, critic, lr=1e-2, gamma=0.5, returns_computation_method=method)


@pytest.mark.parametrize("terminated", [False, True])
@pytest.mark.parametrize("method", ["monte_carlo", "td1"])
def test_targets_bootstrap_only_when_not_terminal(method, terminated):
    """MC propagates last-state value backwards; TD(1) masks terminal bootstrap. @ai-generated"""
    env, episode = make_episode(terminated=terminated)
    trainer = make_trainer(env, method)
    batch = EpisodeBatch([episode], device=trainer.device).for_individual_learners()
    next_values = torch.full_like(batch.rewards, 100.0)
    with patch.object(trainer.critic, "value", return_value=next_values.flatten(0, 1)):
        actual = trainer.compute_returns(batch)
    if method == "td1":
        expected = batch.rewards + trainer.gamma * next_values * (~batch.dones)
    else:
        expected = torch.empty_like(batch.rewards)
        tail = torch.where(batch.dones[-1], torch.zeros_like(next_values[-1]), next_values[-1])
        for t in range(len(episode) - 1, -1, -1):
            tail = batch.rewards[t] + trainer.gamma * tail
            expected[t] = tail
    torch.testing.assert_close(actual, expected)
    assert actual.device == trainer.device
    if terminated:
        torch.testing.assert_close(actual[-1], batch.rewards[-1])
    elif method == "monte_carlo":
        torch.testing.assert_close(actual[-1], batch.rewards[-1] + 50)


@pytest.mark.parametrize("method", ["monte_carlo", "td1"])
def test_update_trains_critic_and_uses_masked_actor(method):
    """Truncated rollouts update both networks without evaluating illegal actions. @ai-generated"""
    env, episode = make_episode(terminated=False)
    trainer = make_trainer(env, method)
    before = [p.detach().clone() for p in trainer.critic.parameters()]
    agent = trainer.make_agent()
    assert agent.actor is trainer.actor
    logs = trainer.update_episode(episode, 1, 2)
    assert np.isfinite(logs["loss"])
    assert np.isfinite(logs["critic_loss"])
    assert any(not torch.equal(p, old) for p, old in zip(trainer.critic.parameters(), before, strict=True))
    batch = EpisodeBatch([episode], device=trainer.device)
    with torch.no_grad():
        dist = trainer.actor.policy(
            batch.obs.flatten(0, 1), batch.extras.flatten(0, 1), available_actions=batch.available_actions.flatten(0, 1)
        )
    assert torch.all(dist.probs[~batch.available_actions.flatten(0, 1)] == 0)


def test_checkpoint_roundtrip_preserves_actor_and_baseline(tmp_path):
    """Save and restore both trainable networks using Trainer's existing machinery. @ai-generated"""
    env, episode = make_episode(terminated=False)
    trainer = make_trainer(env)
    trainer.update_episode(episode, 1, 2)
    trainer.save(tmp_path)
    restored = make_trainer(env)
    restored.load(tmp_path)
    for source, target in ((trainer.actor, restored.actor), (trainer.critic, restored.critic)):
        for old, new in zip(source.parameters(), target.parameters(), strict=True):
            torch.testing.assert_close(new, old)
