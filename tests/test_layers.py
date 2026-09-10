import math

import torch

from marl.nn.layers import AbsLayer, BMMLayer, ILinear, LambdaLayer, NoisyLinear, ReshapeLayer, TransposeLayer


def test_bmm_layer_is_initialised_like_a_linear_layer():
    """`BMMLayer` allocates its parameters with `torch.empty`: they must be initialised."""
    torch.manual_seed(0)
    n_agents, in_features, out_features = 3, 64, 16
    layer = BMMLayer(n_agents, in_features, out_features)
    bound = 1 / math.sqrt(in_features)
    for param in (layer.weight, layer.bias):
        assert torch.isfinite(param).all()
        assert param.abs().max() <= bound
    # Same output scale as the equivalent stack of `torch.nn.Linear`.
    x = torch.randn(128, n_agents, in_features)
    reference = torch.nn.Linear(in_features, out_features)
    ratio = layer.forward(x.transpose(0, 1)).std() / reference.forward(x).std()
    assert 0.5 < ratio < 2.0


def test_bmm_layer_zero_input_features_does_not_crash():
    layer = BMMLayer(2, 0, 8)
    assert torch.isfinite(layer.bias).all()


def test_abs_layer_takes_the_absolute_value():
    x = torch.tensor([-1.0, 2.0, -3.5, 0.0])
    assert torch.equal(AbsLayer().forward(x), torch.abs(x))


def test_reshape_layer_reshapes_to_given_output_shape():
    x = torch.arange(24, dtype=torch.float32)
    layer = ReshapeLayer(2, 3, 4)
    out = layer.forward(x)
    assert out.shape == (2, 3, 4)
    assert torch.equal(out, x.reshape(2, 3, 4))


def test_lambda_layer_applies_the_given_function():
    layer = LambdaLayer(lambda x: x * 2 + 1)
    x = torch.randn(5)
    assert torch.allclose(layer.forward(x), x * 2 + 1)


def test_transpose_layer_swaps_the_given_dimensions():
    x = torch.randn(3, 5, 7)
    layer = TransposeLayer(0, 1)
    out = layer.forward(x)
    assert out.shape == (5, 3, 7)
    assert torch.equal(out, x.transpose(0, 1))


def test_ilinear_matches_stacked_bmm_layers():
    """ILinear without transposition is a thin wrapper around BMMLayer.

    `BMMLayer` batches over its first dimension, which must therefore be the
    agent dimension: (n_agents, batch, in_features).
    """
    torch.manual_seed(0)
    n_agents, in_features, out_features = 4, 8, 6
    layer = ILinear(n_agents, in_features, out_features)
    x = torch.randn(n_agents, 2, in_features)
    bmm_layer: BMMLayer = layer.nn[0]  # type: ignore[assignment]
    assert torch.allclose(layer.forward(x), bmm_layer.forward(x))


def test_ilinear_transpose_in_swaps_batch_and_agent_dims_before_mixing():
    torch.manual_seed(0)
    n_agents, in_features, out_features = 3, 4, 5
    layer = ILinear(n_agents, in_features, out_features, transpose_in=True)
    # Input shaped (batch, n_agents, in_features); transpose_in swaps it to (n_agents, batch, ...).
    x = torch.randn(2, n_agents, in_features)
    out = layer.forward(x)
    assert out.shape == (n_agents, 2, out_features)


def test_ilinear_transpose_out_swaps_back():
    torch.manual_seed(0)
    n_agents, in_features, out_features = 3, 4, 5
    layer = ILinear(n_agents, in_features, out_features, transpose_out=True)
    x = torch.randn(n_agents, 2, in_features)
    out = layer.forward(x)
    assert out.shape == (2, n_agents, out_features)


class TestNoisyLinear:
    def test_output_shape_matches_linear_layer(self):
        layer = NoisyLinear(10, 4)
        x = torch.randn(3, 10)
        assert layer.forward(x).shape == (3, 4)

    def test_training_mode_output_differs_across_calls_due_to_noise(self):
        torch.manual_seed(0)
        layer = NoisyLinear(16, 4)
        layer.train()
        x = torch.randn(2, 16)
        out1 = layer.forward(x)
        out2 = layer.forward(x)
        assert not torch.allclose(out1, out2)

    def test_eval_mode_is_deterministic_and_uses_mu_only(self):
        layer = NoisyLinear(16, 4)
        layer.eval()
        x = torch.randn(2, 16)
        out1 = layer.forward(x)
        out2 = layer.forward(x)
        assert torch.equal(out1, out2)
        expected = torch.nn.functional.linear(x, layer.weight_mu, layer.bias_mu)
        assert torch.equal(out1, expected)

    def test_reset_parameters_bounds(self):
        in_features = 25
        layer = NoisyLinear(in_features, 4)
        bound = 1 / math.sqrt(in_features)
        assert layer.weight_mu.abs().max().item() <= bound
        assert layer.bias_mu.abs().max().item() <= bound
