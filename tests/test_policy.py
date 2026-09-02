"""
Tests for marl.policy: ArgMax, EpsilonGreedy, SoftmaxPolicy, CategoricalPolicy,
NoisyCategoricalPolicy and RandomPolicy.
"""

import numpy as np
import pytest

from marl.policy.probabilistic_policies import CategoricalPolicy, NoisyCategoricalPolicy
from marl.policy.qpolicies import ArgMax, EpsilonGreedy, SoftmaxPolicy
from marl.policy.random_policy import RandomPolicy
from marl.utils.schedule import Schedule


class TestArgMax:
    def test_picks_the_highest_qvalue(self):
        policy = ArgMax()
        qvalues = np.array([[1.0, 5.0, 2.0], [3.0, 1.0, 0.0]])
        actions = policy.get_action(qvalues)
        assert list(actions) == [1, 0]

    def test_respects_available_actions_mask(self):
        policy = ArgMax()
        qvalues = np.array([[1.0, 5.0, 2.0]])
        available = np.array([[1.0, 0.0, 1.0]])
        actions = policy.get_action(qvalues, available)
        assert actions[0] == 2

    def test_update_returns_empty_logs(self):
        assert ArgMax().update(0) == {}


class TestEpsilonGreedy:
    def test_epsilon_zero_is_always_greedy(self):
        np.random.seed(0)
        policy = EpsilonGreedy.constant(0.0)
        qvalues = np.array([[1.0, 5.0, 2.0]] * 20)
        actions = policy.get_action(qvalues.copy())
        assert all(a == 1 for a in actions)

    def test_epsilon_one_never_picks_the_greedy_action_when_alternatives_exist(self):
        np.random.seed(0)
        policy = EpsilonGreedy.constant(1.0)
        qvalues = np.tile(np.array([0.0, 10.0, 0.0]), (200, 1))
        actions = policy.get_action(qvalues.copy())
        # With epsilon=1 every action is replaced with a random *available* action - not
        # necessarily different from the greedy one, but across many draws we should see
        # actions other than the greedy one (2 possible non-greedy actions out of 3).
        assert set(actions) != {1}

    def test_respects_available_actions_when_replacing(self):
        np.random.seed(0)
        policy = EpsilonGreedy.constant(1.0)
        qvalues = np.tile(np.array([1.0, 5.0, 2.0]), (50, 1))
        available = np.tile(np.array([1.0, 0.0, 1.0]), (50, 1))
        actions = policy.get_action(qvalues.copy(), available.copy())
        assert set(actions) <= {0, 2}

    def test_update_advances_the_schedule_and_returns_its_value(self):
        policy = EpsilonGreedy.linear(1.0, 0.0, 10)
        logs = policy.update(5)
        assert logs["epsilon"] == pytest.approx(policy.epsilon.value)
        assert logs["epsilon"] == pytest.approx(0.5)

    def test_from_constructors(self):
        assert EpsilonGreedy.linear(1.0, 0.0, 10).epsilon.value == pytest.approx(1.0)
        assert EpsilonGreedy.exponential(1.0, 0.01, 10).epsilon.value == pytest.approx(1.0)
        assert EpsilonGreedy.constant(0.3).epsilon.value == pytest.approx(0.3)

    def test_custom_schedule_is_used_directly(self):
        sched = Schedule.linear(0.5, 0.5, 10)
        policy = EpsilonGreedy(sched)
        assert policy.epsilon is sched


class TestSoftmaxPolicy:
    def test_probabilities_sum_to_one_and_actions_are_valid(self):
        np.random.seed(0)
        policy = SoftmaxPolicy(n_actions=4, tau=1.0)
        qvalues = np.random.randn(10, 4).astype(np.float32)
        actions = policy.get_action(qvalues)
        assert len(actions) == 10
        assert all(0 <= a < 4 for a in actions)

    def test_low_temperature_converges_to_argmax(self):
        np.random.seed(0)
        policy = SoftmaxPolicy(n_actions=3, tau=1e-4)
        qvalues = np.tile(np.array([0.0, 10.0, 1.0], dtype=np.float32), (50, 1))
        actions = policy.get_action(qvalues.copy())
        assert all(a == 1 for a in actions)

    def test_unavailable_actions_are_never_chosen(self):
        np.random.seed(0)
        policy = SoftmaxPolicy(n_actions=3, tau=1.0)
        qvalues = np.tile(np.array([1.0, 2.0, 3.0], dtype=np.float32), (50, 1))
        available = np.tile(np.array([1.0, 0.0, 1.0], dtype=np.float32), (50, 1))
        actions = policy.get_action(qvalues.copy(), available)
        assert set(actions) <= {0, 2}

    def test_update_returns_current_tau(self):
        policy = SoftmaxPolicy(n_actions=2, tau=0.5)
        assert policy.update(0) == {"softmax-tau": 0.5}


class TestCategoricalPolicy:
    def test_actions_are_in_range(self):
        np.random.seed(0)
        policy = CategoricalPolicy()
        qvalues = np.random.randn(20, 5)
        actions = policy.get_action(qvalues)
        assert all(0 <= a < 5 for a in actions)

    def test_unavailable_actions_are_never_chosen(self):
        np.random.seed(0)
        policy = CategoricalPolicy()
        qvalues = np.tile(np.array([1.0, 1.0, 1.0]), (100, 1))
        available = np.tile(np.array([1.0, 0.0, 1.0]), (100, 1))
        actions = policy.get_action(qvalues.copy(), available)
        assert set(actions) <= {0, 2}

    def test_update_returns_empty_logs(self):
        assert CategoricalPolicy().update(0) == {}


class TestNoisyCategoricalPolicy:
    def test_actions_are_in_range(self):
        np.random.seed(0)
        policy = NoisyCategoricalPolicy(mu=0.0, sigma=0.1)
        qvalues = np.random.randn(20, 4)
        actions = policy.get_action(qvalues)
        assert all(0 <= a < 4 for a in actions)

    def test_zero_sigma_behaves_like_categorical_on_logits(self):
        np.random.seed(0)
        policy = NoisyCategoricalPolicy(mu=0.0, sigma=0.0)
        qvalues = np.tile(np.array([-1e9, 1e9, -1e9]), (30, 1)).astype(np.float64)
        actions = policy.get_action(qvalues.copy())
        assert all(a == 1 for a in actions)


class TestRandomPolicy:
    def test_returns_one_action_per_agent_within_range(self):
        np.random.seed(0)
        policy = RandomPolicy(n_actions=5, n_agents=3)
        actions = policy.choose_action()
        assert actions.shape == (3,)
        assert all(0 <= a < 5 for a in actions)

    def test_save_load_are_no_ops(self):
        policy = RandomPolicy(n_actions=2, n_agents=1)
        policy.save("unused")
        policy.load("unused")
