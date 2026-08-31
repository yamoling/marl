import math

import torch

from marl.nn.layers import BMMLayer


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
