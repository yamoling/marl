"""
Tests for marl.models.nn.actor_critic: Actor/Critic base classes, CategoricalActor
and its `to_one_hot()` conversion.

`to_one_hot()` builds a nested `DiscreteOneHotActor` class whose `__init__` used to
read `self.n_actions` / `self.obs_shape` / ... before those attributes existed on the
new instance (it should have read them from the wrapped `actor` argument instead).
This is a regression test for that bug.
"""

import torch
from marlenv.catalog import DiscreteMockEnv

from marl.env import EnvConfig
from marl.nn.model_bank import actor_critics


def _make_actor():
    env = DiscreteMockEnv()
    env_config = EnvConfig.from_any(env)
    actor, _ = actor_critics.from_env(env_config, recurrent=False)
    return actor, env


class TestCategoricalActor:
    def test_policy_returns_categorical_distribution(self):
        actor, env = _make_actor()
        obs = torch.randn(2, env.n_agents, actor.obs_size)
        extras = torch.randn(2, env.n_agents, actor.extras_size)
        dist = actor.policy(obs, extras)
        assert isinstance(dist, torch.distributions.Categorical)
        assert dist.probs.shape == (2, env.n_agents, env.n_actions)

    def test_mask_replaces_unavailable_logits_with_negative_infinity(self):
        actor, env = _make_actor()
        x = torch.zeros(1, env.n_agents, env.n_actions)
        available = torch.ones_like(x, dtype=torch.bool)
        available[..., 0] = False
        masked = actor.mask(x, available)
        assert torch.all(masked[..., 0] == -torch.inf)
        assert torch.all(masked[..., 1:] == 0.0)

    def test_mask_is_identity_when_no_available_actions_given(self):
        actor, _ = _make_actor()
        x = torch.randn(3, 4)
        assert torch.equal(actor.mask(x, None), x)

    def test_log_probs_matches_manual_categorical_computation(self):
        actor, env = _make_actor()
        obs = torch.randn(2, env.n_agents, actor.obs_size)
        extras = torch.randn(2, env.n_agents, actor.extras_size)
        actions = torch.randint(0, env.n_actions, (2, env.n_agents))
        log_probs = actor.log_probs(obs, extras, actions)
        dist = actor.policy(obs, extras)
        assert torch.allclose(log_probs, dist.log_prob(actions))


class TestToOneHot:
    def test_returns_one_hot_categorical_distribution(self):
        actor, env = _make_actor()
        one_hot_actor = actor.to_one_hot()
        obs = torch.randn(2, env.n_agents, actor.obs_size)
        extras = torch.randn(2, env.n_agents, actor.extras_size)
        dist = one_hot_actor.policy(obs, extras)
        assert isinstance(dist, torch.distributions.OneHotCategorical)

    def test_preserves_shape_metadata_from_wrapped_actor(self):
        actor, _ = _make_actor()
        one_hot_actor = actor.to_one_hot()
        assert one_hot_actor.n_actions == actor.n_actions
        assert one_hot_actor.obs_shape == actor.obs_shape
        assert one_hot_actor.extras_shape == actor.extras_shape
        assert one_hot_actor.n_agents == actor.n_agents

    def test_samples_are_valid_one_hot_vectors(self):
        actor, env = _make_actor()
        one_hot_actor = actor.to_one_hot()
        obs = torch.randn(1, env.n_agents, actor.obs_size)
        extras = torch.randn(1, env.n_agents, actor.extras_size)
        sample = one_hot_actor.policy(obs, extras).sample()
        assert sample.shape == (1, env.n_agents, env.n_actions)
        assert torch.all(sample.sum(dim=-1) == 1.0)


class TestQNetworkToSoftmaxActor:
    """`QNetwork.to_softmax_actor()` has the same nested-class attribute bug; covered here too."""

    def test_preserves_shape_metadata_from_wrapped_qnetwork(self):
        from marl.nn.model_bank import qnetworks

        env = DiscreteMockEnv()
        qnet = qnetworks.from_env(env, hidden_sizes=(8,))
        actor = qnet.to_softmax_actor()
        assert actor.n_actions == qnet.n_actions
        assert actor.obs_shape == qnet.obs_shape
        assert actor.extras_shape == qnet.extras_shape


class TestCritic:
    def test_value_shape_matches_agents(self):
        from marl.nn.model_bank.actor_critics import critics

        env = DiscreteMockEnv()
        critic = critics.from_env(env, independent=False)
        obs = torch.randn(3, env.n_agents, critic.obs_size)
        extras = torch.randn(3, env.n_agents, 0)
        value = critic.value(obs, extras)
        assert value.shape[-1] == 1 or value.shape == (3, env.n_agents)
