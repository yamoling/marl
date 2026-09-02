"""Tests for marl.algos.optimism.vbe.VBE (Value Bonuses using Ensemble)."""

import math

import numpy as np
import torch
from marlenv import Transition
from marlenv.catalog import DiscreteMockEnv

from marl.algos.optimism.vbe import VBE
from marl.models.batch import TransitionBatch
from marl.nn.model_bank import qnetworks


def _make_vbe(n=3, lr=1e-3):
    env = DiscreteMockEnv()
    qnet = qnetworks.from_env(env, hidden_sizes=(8,), duelling=False)
    return VBE(qnet, n=n, lr=lr), env


class TestVBEInit:
    def test_creates_n_rqfs_and_targets(self):
        vbe, _ = _make_vbe(n=4)
        assert len(vbe._rqfs) == 4
        assert len(vbe._target_rqfs) == 4
        assert len(vbe._optimizers) == 4

    def test_rqf_and_target_are_distinct_objects_with_different_weights(self):
        torch.manual_seed(0)
        vbe, _ = _make_vbe(n=1)
        rqf_params = list(vbe._rqfs[0].parameters())
        target_params = list(vbe._target_rqfs[0].parameters())
        assert vbe._rqfs[0] is not vbe._target_rqfs[0]
        # Each is separately randomised, so at least one parameter tensor should differ.
        assert any(not torch.equal(a, b) for a, b in zip(rqf_params, target_params))


class TestComputeBonus:
    def test_bonus_shape_matches_agents_and_actions(self):
        vbe, env = _make_vbe(n=2)
        obs, _ = env.reset()
        bonus = vbe.compute_bonus(obs)
        assert bonus.shape == (env.n_agents, env.n_actions)

    def test_bonus_is_non_negative(self):
        """The bonus is defined as `abs(target - predicted)`, hence never negative."""
        vbe, env = _make_vbe(n=2)
        obs, _ = env.reset()
        bonus = vbe.compute_bonus(obs)
        assert np.all(bonus >= 0)

    def test_bonus_is_zero_when_rqf_and_target_share_weights(self):
        vbe, env = _make_vbe(n=1)
        vbe._target_rqfs[0].load_state_dict(vbe._rqfs[0].state_dict())
        obs, _ = env.reset()
        bonus = vbe.compute_bonus(obs)
        assert np.allclose(bonus, 0.0, atol=1e-5)

    def test_bonus_is_appended_to_history(self):
        vbe, env = _make_vbe(n=2)
        obs, _ = env.reset()
        assert len(vbe._bonus_history) == 0
        vbe.compute_bonus(obs)
        vbe.compute_bonus(obs)
        assert len(vbe._bonus_history) == 2


class TestUpdate:
    def _make_batch(self, env, n_transitions=4):
        obs, state = env.reset()
        transitions = []
        for _ in range(n_transitions):
            action = env.sample_action()
            step = env.step(action)
            transitions.append(Transition.from_step(obs, state, action, step))
            obs, state = step.obs, step.state
        return TransitionBatch(transitions)

    def test_update_requires_a_prior_compute_bonus_call(self):
        """`update` reduces `_bonus_history` with np.stack, which fails on an empty list."""
        import pytest

        vbe, env = _make_vbe(n=2)
        batch = self._make_batch(env)
        with pytest.raises(ValueError):
            vbe.update(batch)

    def test_update_returns_finite_logs_and_clears_history(self):
        vbe, env = _make_vbe(n=2, lr=1e-2)
        for _ in range(4):
            obs, _ = env.reset()
            vbe.compute_bonus(obs)
        batch = self._make_batch(env)
        logs = vbe.update(batch)
        assert set(logs.keys()) == {"vbe_loss", "mean_vbe_bonus"}
        assert all(math.isfinite(v) for v in logs.values())
        assert len(vbe._bonus_history) == 0

    def test_update_changes_the_selected_rqfs_parameters(self):
        torch.manual_seed(0)
        vbe, env = _make_vbe(n=1, lr=1e-1)
        obs, _ = env.reset()
        vbe.compute_bonus(obs)
        batch = self._make_batch(env)
        before = [p.clone() for p in vbe._rqfs[0].parameters()]
        vbe.update(batch)
        after = list(vbe._rqfs[0].parameters())
        assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_to_moves_all_rqfs_and_targets():
    vbe, _ = _make_vbe(n=2)
    vbe.to(torch.device("cpu"))
    assert vbe._device.type == "cpu"
    for rqf in vbe._rqfs + vbe._target_rqfs:
        assert rqf.device.type == "cpu"
