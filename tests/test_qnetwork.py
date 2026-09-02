"""Tests for marl.models.nn.qnetwork.QNetwork: duelling head, shapes, from_env, to_softmax_actor."""

import pytest
import torch
from marlenv.catalog import DiscreteMockEnv

from marl.nn.model_bank import qnetworks


def _make(duelling: bool, n_objectives: int = 1, **kwargs):
    env = DiscreteMockEnv()
    return qnetworks.from_env(env, hidden_sizes=(8,), duelling=duelling, n_objectives=n_objectives, **kwargs)


class TestDuellingHead:
    def test_output_shape_has_one_extra_unit_for_value_stream(self):
        net = _make(duelling=True)
        assert net.output_shape == (net.n_actions + 1,)

    def test_non_duelling_output_shape_matches_n_actions(self):
        net = _make(duelling=False)
        assert net.output_shape == (net.n_actions,)

    def test_get_qvalues_decomposition_matches_manual_formula(self):
        net = _make(duelling=True)
        raw = torch.randn(5, net.n_agents, net.n_actions + 1)
        qvalues = net._get_qvalues(raw)
        value = raw[..., -1:].expand(*raw.shape[:-1], net.n_actions)
        adv = raw[..., :-1]
        expected = value + adv - adv.mean(dim=-1, keepdim=True)
        assert torch.allclose(qvalues, expected)

    def test_get_qvalues_is_identity_when_not_duelling(self):
        net = _make(duelling=False)
        raw = torch.randn(5, net.n_agents, net.n_actions)
        assert torch.equal(net._get_qvalues(raw), raw)

    def test_advantages_are_mean_centred_around_value(self):
        """A defining property of duelling DQN: mean(qvalues - value) == 0 across actions."""
        net = _make(duelling=True)
        raw = torch.randn(3, net.n_agents, net.n_actions + 1)
        qvalues = net._get_qvalues(raw)
        value = raw[..., -1]
        centred_mean = (qvalues - value.unsqueeze(-1)).mean(dim=-1)
        assert torch.allclose(centred_mean, torch.zeros_like(centred_mean), atol=1e-6)

    def test_duelling_with_multi_objective_raises(self):
        with pytest.raises(NotImplementedError):
            _make(duelling=True, n_objectives=2)


class TestQNetworkShapesAndProperties:
    def test_obs_size_is_product_of_obs_shape(self):
        import math

        net = _make(duelling=False)
        assert net.obs_size == math.prod(net.obs_shape)

    def test_is_multi_objective_flag(self):
        assert _make(duelling=False, n_objectives=1).is_multi_objective is False
        assert _make(duelling=False, n_objectives=2).is_multi_objective is True

    def test_qvalues_single_observation_shape(self):
        env = DiscreteMockEnv()
        net = _make(duelling=True)
        obs, _ = env.reset()
        q = net.qvalues(obs)
        assert q.shape == (env.n_agents, env.n_actions)

    def test_batch_qvalues_shape(self):
        net = _make(duelling=True)
        obs = torch.randn(7, net.n_agents, net.obs_size)
        extras = torch.randn(7, net.n_agents, 0)
        q = net.batch_qvalues(obs, extras)
        assert q.shape == (7, net.n_agents, net.n_actions)

    def test_from_env_infers_shapes(self):
        env = DiscreteMockEnv()
        net = qnetworks.from_env(env)
        assert net.n_actions == env.n_actions
        assert net.n_agents == env.n_agents
        assert net.obs_shape == env.observation_shape


class TestToSoftmaxActor:
    def test_produces_logits_matching_qvalues(self):
        net = _make(duelling=True)
        actor = net.to_softmax_actor()
        obs = torch.randn(2, net.n_agents, net.obs_size)
        extras = torch.randn(2, net.n_agents, 0)
        logits = actor.logits(obs, extras)
        assert torch.allclose(logits, net.batch_qvalues(obs, extras))

    def test_masks_unavailable_actions_with_negative_infinity(self):
        net = _make(duelling=True)
        obs = torch.randn(1, net.n_agents, net.obs_size)
        extras = torch.randn(1, net.n_agents, 0)
        actor = net.to_softmax_actor()
        available = torch.ones(1, net.n_agents, net.n_actions, dtype=torch.bool)
        available[..., 0] = False
        logits = actor.logits(obs, extras, available_actions=available)
        assert torch.all(logits[..., 0] == -torch.inf)
