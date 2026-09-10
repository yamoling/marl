"""Tests for marl.utils.stats.running_mean_std.RunningMeanStd."""

import torch

from marl.utils.stats.running_mean_std import RunningMeanStd


def test_starts_with_zero_mean_and_unit_variance():
    rms = RunningMeanStd(shape=(3,))
    assert torch.allclose(rms.mean, torch.zeros(3))
    assert torch.allclose(rms.variance, torch.ones(3))
    assert torch.allclose(rms.std, torch.ones(3))


def test_update_matches_batch_statistics_on_first_call():
    torch.manual_seed(0)
    rms = RunningMeanStd(shape=(4,))
    batch = torch.randn(1000, 4) * 2.0 + 3.0
    rms.update(batch)
    assert torch.allclose(rms.mean, batch.mean(dim=0), atol=1e-4)
    assert torch.allclose(rms.variance, batch.var(dim=0, unbiased=False), atol=1e-4)
    assert rms.count == 1000


def test_sequential_updates_match_combined_statistics():
    torch.manual_seed(1)
    full_batch = torch.randn(500, 2) * 3.0 - 1.0
    rms = RunningMeanStd(shape=(2,))
    rms.update(full_batch[:200])
    rms.update(full_batch[200:])
    assert torch.allclose(rms.mean, full_batch.mean(dim=0), atol=1e-3)
    assert torch.allclose(rms.variance, full_batch.var(dim=0, unbiased=False), atol=1e-3)
    assert rms.count == 500


def test_normalise_updates_statistics_by_default():
    torch.manual_seed(2)
    rms = RunningMeanStd(shape=(1,))
    batch = torch.randn(100, 1)
    rms.normalise(batch)
    assert rms.count == 100


def test_normalise_without_update_leaves_statistics_untouched():
    rms = RunningMeanStd(shape=(1,))
    rms.update(torch.randn(50, 1))
    count_before = rms.count
    rms.normalise(torch.randn(10, 1), update=False)
    assert rms.count == count_before


def test_normalise_clips_to_bounds():
    rms = RunningMeanStd(shape=(1,), clip_min=-2.0, clip_max=2.0)
    # Force statistics so the raw normalised value would be huge.
    rms.mean = torch.zeros(1)
    rms.variance = torch.full((1,), 1e-6)
    out = rms.normalise(torch.tensor([[1000.0]]), update=False)
    assert torch.all(out <= 2.0)
    assert torch.all(out >= -2.0)


def test_to_moves_mean_and_variance_to_device():
    rms = RunningMeanStd(shape=(2,))
    rms.to(torch.device("cpu"))
    assert rms.mean.device.type == "cpu"
    assert rms.variance.device.type == "cpu"
