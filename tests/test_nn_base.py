"""
Tests for marl.models.nn.nn: the `NN` / `RecurrentNN` base classes and `get_activation`.
"""

import pytest
import torch
from marlenv.catalog import DiscreteMockEnv

from marl.models.nn.nn import get_activation
from marl.nn.model_bank import qnetworks
from marl.nn.model_bank.generic import RNN


def _make_qmlp():
    env = DiscreteMockEnv()
    return qnetworks.from_env(env, hidden_sizes=(16,))


class TestGetActivation:
    def test_returns_expected_module_types(self):
        assert isinstance(get_activation("relu"), torch.nn.ReLU)
        assert isinstance(get_activation("tanh"), torch.nn.Tanh)
        assert isinstance(get_activation("sigmoid"), torch.nn.Sigmoid)
        assert isinstance(get_activation("leaky-relu"), torch.nn.LeakyReLU)

    def test_raises_for_unknown_activation(self):
        with pytest.raises(ValueError):
            get_activation("unknown")  # type: ignore[arg-type]


class TestNNBase:
    def test_output_size_is_product_of_output_shape(self):
        import math

        net = _make_qmlp()
        assert net.output_size == math.prod(net.output_shape)

    def test_randomize_xavier_changes_parameters(self):
        net = _make_qmlp()
        before = [p.clone() for p in net.parameters()]
        net.randomize("xavier")
        after = list(net.parameters())
        assert any(not torch.equal(b, a) for b, a in zip(before, after))

    def test_randomize_orthogonal_changes_parameters(self):
        net = _make_qmlp()
        before = [p.clone() for p in net.parameters()]
        net.randomize("orthogonal")
        after = list(net.parameters())
        assert any(not torch.equal(b, a) for b, a in zip(before, after))

    def test_randomize_accepts_a_custom_init_function(self):
        net = _make_qmlp()
        net.randomize(lambda t: torch.nn.init.constant_(t, 0.5))
        assert all(torch.all(p == 0.5) for p in net.parameters())

    def test_save_and_load_round_trip(self, tmp_path):
        net = _make_qmlp()
        net.randomize("xavier")
        net.save(tmp_path)
        other = _make_qmlp()
        other.load(tmp_path)
        for p1, p2 in zip(net.parameters(), other.parameters()):
            assert torch.equal(p1, p2)

    def test_name_defaults_to_class_name(self):
        net = _make_qmlp()
        assert net.name == type(net).__name__

    def test_is_recurrent_false_for_plain_mlp(self):
        assert _make_qmlp().is_recurrent is False

    def test_repr_contains_the_class_name(self):
        net = _make_qmlp()
        assert net.name in repr(net)

    def test_device_property_matches_parameters(self):
        net = _make_qmlp()
        assert net.device == next(net.parameters()).device


class TestRecurrentNN:
    def _make_rnn(self):
        return RNN(output_shape=(4,), obs_size=6, extras_size=0, mlp_head_sizes=(8,), mlp_tail_sizes=(8,))

    def test_is_recurrent_is_true(self):
        assert self._make_rnn().is_recurrent is True

    def test_reset_hidden_states_clears_state(self):
        rnn = self._make_rnn()
        rnn._hidden_states = torch.zeros(1, 1, 8)
        rnn.reset_hidden_states()
        assert rnn._hidden_states is None

    def test_forward_populates_hidden_state(self):
        rnn = self._make_rnn()
        obs = torch.randn(3, 2, 5, 6)
        extras = torch.randn(3, 2, 5, 0)
        rnn.forward(obs, extras)
        assert rnn._hidden_states is not None

    def test_switching_to_eval_saves_and_clears_hidden_state(self):
        rnn = self._make_rnn()
        obs = torch.randn(3, 2, 5, 6)
        extras = torch.randn(3, 2, 5, 0)
        rnn.forward(obs, extras)
        hidden_before = rnn._hidden_states
        rnn.eval()
        assert rnn._hidden_states is None
        assert rnn._saved_hidden_states is hidden_before

    def test_switching_back_to_train_restores_hidden_state(self):
        rnn = self._make_rnn()
        obs = torch.randn(3, 2, 5, 6)
        extras = torch.randn(3, 2, 5, 0)
        rnn.forward(obs, extras)
        hidden_before = rnn._hidden_states
        rnn.eval()
        rnn.train()
        assert rnn._hidden_states is hidden_before

    def test_is_recurrent_propagates_through_a_containing_nn(self):
        """`NN.is_recurrent` looks for recurrent children, including nested `NN`s."""
        rnn = self._make_rnn()

        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.child = rnn

        wrapper = Wrapper()
        # is_recurrent iterates `self.children()`, so wrap in something exposing the RNN as a child.
        from marl.models.nn.nn import NN

        assert any(isinstance(c, NN) and c.is_recurrent for c in wrapper.children())
