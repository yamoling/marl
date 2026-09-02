"""Tests for marl.nn.model_bank.generic: MLP and CNN."""

import torch

from marl.nn.model_bank.generic import CNN, MLP


class TestMLP:
    def test_forward_shape_matches_output_shape(self):
        mlp = MLP((4,), obs_size=10, extras_size=2, hidden_sizes=(16, 8))
        obs = torch.randn(5, 3, 10)
        extras = torch.randn(5, 3, 2)
        out = mlp.forward(obs, extras)
        assert out.shape == (5, 3, 4)

    def test_input_size_accounts_for_extras(self):
        mlp = MLP((4,), obs_size=10, extras_size=2)
        assert mlp.input_size == 12

    def test_layer_sizes_starts_and_ends_correctly(self):
        mlp = MLP((4,), obs_size=10, extras_size=0, hidden_sizes=(16, 8))
        assert mlp.layer_sizes == (10, 16, 8, 4)

    def test_independent_mode_requires_n_agents(self):
        import pytest

        with pytest.raises(AssertionError):
            MLP((4,), obs_size=10, extras_size=0, independent=True, n_agents=-1)

    def test_independent_mode_forward_shape(self):
        n_agents = 3
        mlp = MLP((4,), obs_size=10, extras_size=0, hidden_sizes=(8,), independent=True, n_agents=n_agents)
        obs = torch.randn(5, n_agents, 10)
        extras = torch.randn(5, n_agents, 0)
        out = mlp.forward(obs, extras)
        assert out.shape == (5, n_agents, 4)

    def test_output_activation_adds_final_activation_layer(self):
        with_activation = MLP((4,), obs_size=10, extras_size=0, hidden_sizes=(8,), output_activation="relu")
        without_activation = MLP((4,), obs_size=10, extras_size=0, hidden_sizes=(8,), output_activation=None)
        assert len(with_activation.nn) > len(without_activation.nn)

    def test_shared_weights_agents_produce_identical_outputs(self):
        """Non-independent MLPs share weights across agents: identical inputs give identical outputs."""
        torch.manual_seed(0)
        mlp = MLP((4,), obs_size=6, extras_size=0, hidden_sizes=(8,))
        x = torch.randn(1, 6)
        obs = x.unsqueeze(1).repeat(1, 3, 1)
        extras = torch.zeros(1, 3, 0)
        out = mlp.forward(obs, extras)
        assert torch.allclose(out[0, 0], out[0, 1])
        assert torch.allclose(out[0, 0], out[0, 2])


class TestCNN:
    def test_forward_output_shape(self):
        torch.manual_seed(0)
        cnn = CNN((3, 8, 8), filters=(4, 4), kernel_sizes=(3, 3), strides=(1, 1))
        obs = torch.randn(2, 5, 3, 8, 8)  # (batch, n_agents, C, H, W)
        out = cnn.forward(obs)
        assert out.shape == (2, 5, cnn.output_size)

    def test_handles_episode_shaped_input(self):
        """CNN forward also supports (time, batch, n_agents, C, H, W) inputs."""
        torch.manual_seed(0)
        cnn = CNN((3, 8, 8), filters=(4,), kernel_sizes=(3,), strides=(1,))
        obs = torch.randn(4, 2, 5, 3, 8, 8)
        out = cnn.forward(obs)
        assert out.shape == (4, 2, 5, cnn.output_size)

    def test_raises_on_mismatched_hyperparameter_lengths(self):
        import pytest

        with pytest.raises(AssertionError):
            CNN((3, 8, 8), filters=(4, 4), kernel_sizes=(3,), strides=(1, 1))
