"""
Tests for marl.models.agent.agent.Agent (the base class), AgentWrapper, RandomAgent,
RandomOneHot, the ReplayAgent family and the bandit module.
"""

import numpy as np
import pytest
import torch
from marlenv.catalog import DiscreteMockEnv

from marl.agents.random_agent import RandomAgent, RandomOneHot
from marl.agents.replay_agent import CombinedReplayAgent, ReplayActionsOnlyAgent, ReplayAgent, SimpleReplayAgent
from marl.models.action import Action
from marl.models.agent.agent import Agent
from marl.models.agent.agent_wrapper import AgentWrapper
from marl.models.agent.bandit import CategoricalBandit
from marl.nn.model_bank import qnetworks


class _DummyAgent(Agent):
    """Minimal concrete Agent exposing a single QMLP network, for exercising the base class."""

    def __init__(self):
        super().__init__()
        env = DiscreteMockEnv()
        self.qnetwork = qnetworks.from_env(env, hidden_sizes=(8,))

    def choose_action(self, observation, *, with_details: bool = False):
        return Action(np.zeros(4, dtype=np.int64))


class TestAgentBase:
    def test_starts_in_training_mode_on_cpu(self):
        agent = _DummyAgent()
        assert agent.is_training
        assert not agent.is_testing
        assert agent.device.type == "cpu"

    def test_networks_lists_nn_attributes(self):
        agent = _DummyAgent()
        assert agent.qnetwork in agent.networks()

    def test_recurrent_networks_is_empty_for_mlp_only_agent(self):
        agent = _DummyAgent()
        assert agent.recurrent_networks == []

    def test_set_testing_flips_flags_and_network_mode(self):
        agent = _DummyAgent()
        agent.set_testing()
        assert agent.is_testing
        assert not agent.qnetwork.training

    def test_set_training_flips_flags_and_network_mode(self):
        agent = _DummyAgent()
        agent.set_testing()
        agent.set_training()
        assert agent.is_training
        assert agent.qnetwork.training

    def test_randomize_changes_network_parameters(self):
        agent = _DummyAgent()
        before = [p.clone() for p in agent.qnetwork.parameters()]
        agent.randomize()
        after = list(agent.qnetwork.parameters())
        assert any(not torch.equal(b, a) for b, a in zip(before, after))

    def test_to_updates_device_and_moves_networks(self):
        agent = _DummyAgent()
        device = torch.device("cpu")
        result = agent.to(device)
        assert result is agent
        assert agent.device == device
        assert agent.qnetwork.device == device

    def test_can_autosave_when_network_names_are_unique(self):
        agent = _DummyAgent()
        assert agent._can_autosave()

    def test_save_then_load_round_trips_parameters(self, tmp_path):
        agent = _DummyAgent()
        agent.randomize()
        agent.save(tmp_path)
        other = _DummyAgent()
        other.load(tmp_path)
        for p1, p2 in zip(agent.qnetwork.parameters(), other.qnetwork.parameters()):
            assert torch.equal(p1, p2)

    def test_new_episode_resets_recurrent_networks_only(self):
        agent = _DummyAgent()
        # No recurrent networks: should simply not raise.
        agent.new_episode()

    def test_seed_is_reproducible(self):
        agent = _DummyAgent()
        agent.seed(123)
        a = np.random.rand(3)
        agent.seed(123)
        b = np.random.rand(3)
        assert np.array_equal(a, b)


class TestAgentWrapper:
    def test_choose_action_delegates_to_wrapped_agent(self):
        env = DiscreteMockEnv()
        inner = RandomAgent(env)
        wrapper = AgentWrapper(inner)
        obs, _ = env.reset()
        action = wrapper.choose_action(obs)
        assert isinstance(action, Action)

    def test_networks_include_wrapper_and_wrapped_networks(self):
        agent = _DummyAgent()
        wrapper = AgentWrapper(agent)
        nets = wrapper.networks()
        assert agent.qnetwork in nets

    def test_to_moves_both_wrapper_and_wrapped_agent(self):
        agent = _DummyAgent()
        wrapper = AgentWrapper(agent)
        wrapper.to(torch.device("cpu"))
        assert agent.device.type == "cpu"
        assert wrapper.device.type == "cpu"

    def test_set_testing_propagates_to_wrapped_agent(self):
        agent = _DummyAgent()
        wrapper = AgentWrapper(agent)
        wrapper.set_testing()
        assert agent.is_testing
        assert wrapper.is_testing


class TestRandomAgent:
    def test_action_is_within_available_actions(self):
        env = DiscreteMockEnv()
        agent = RandomAgent(env)
        obs, _ = env.reset()
        for _ in range(20):
            action = agent.choose_action(obs)
            assert action.action.shape == (env.n_agents,)
            assert np.all(action.action >= 0)
            assert np.all(action.action < env.n_actions)

    def test_value_is_always_zero(self):
        agent = RandomAgent(DiscreteMockEnv())
        assert agent.value(None) == 0.0

    def test_save_load_to_are_no_ops(self):
        agent = RandomAgent(DiscreteMockEnv())
        agent.save("unused")
        agent.load("unused")
        assert agent.to("cpu") is agent


class TestRandomOneHot:
    def test_returns_one_hot_actions_for_every_agent(self):
        agent = RandomOneHot(n_actions=4, n_agents=3)
        env = DiscreteMockEnv()
        obs, _ = env.reset()
        action = agent.choose_action(obs)
        assert action.action.shape == (3, 4)
        assert np.all(action.action.sum(axis=1) == 1)


class TestReplayAgents:
    def test_replay_actions_only_agent_returns_stored_actions_in_order(self):
        stored = np.array([[1, 2], [3, 4], [5, 6]])
        agent = ReplayAgent.from_actions_only(stored)
        assert isinstance(agent, ReplayActionsOnlyAgent)
        for expected in stored:
            action = agent.choose_action(None)
            assert np.array_equal(action.action, expected)

    def test_combined_replay_agent_flags_mismatch(self):
        env = DiscreteMockEnv()
        obs, _ = env.reset()
        wrapped = RandomAgent(env)
        stored_actions = np.array([[999, 999, 999, 999]])  # guaranteed to differ from a random action
        combined = CombinedReplayAgent(stored_actions, wrapped)
        action = combined.choose_action(obs)
        assert combined.mismatch
        assert len(combined.mismatch_details) == 1
        # The mismatch is resolved in favour of the stored (replayed) action.
        assert np.array_equal(action.action, stored_actions[0])

    def test_combined_replay_agent_no_mismatch_when_actions_match(self):
        env = DiscreteMockEnv()
        obs, _ = env.reset()

        class _FixedAgent(Agent):
            def choose_action(self, observation, *, with_details=False):
                return Action(np.array([0, 0, 0, 0]))

        combined = CombinedReplayAgent(np.array([[0, 0, 0, 0]]), _FixedAgent())
        combined.choose_action(obs)
        assert not combined.mismatch

    def test_simple_replay_agent_loads_weights_and_delegates(self, tmp_path):
        agent = _DummyAgent()
        agent.save(tmp_path)
        wrapped = _DummyAgent()
        replay = SimpleReplayAgent(wrapped, tmp_path)
        obs, _ = DiscreteMockEnv().reset()
        action = replay.choose_action(obs)
        assert isinstance(action, Action)


class _FixedBandit(CategoricalBandit):
    def __init__(self, n_actions: int, fixed_action: int):
        super().__init__(n_actions)
        self._fixed_action = fixed_action

    def choose_action(self, /, **kwargs):
        return self._fixed_action


class TestCategoricalBandit:
    def test_to_one_hot_wraps_the_bandit(self):
        bandit = _FixedBandit(n_actions=5, fixed_action=0)
        one_hot = bandit.to_one_hot()
        assert one_hot.n_actions == 5

    def test_one_hot_bandit_produces_valid_one_hot_vector(self):
        one_hot = _FixedBandit(n_actions=4, fixed_action=2).to_one_hot()
        vec = one_hot.choose_action()
        assert vec.shape == (4,)
        assert vec[2] == 1.0
        assert vec.sum() == 1.0
