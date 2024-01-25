import torch
from marl.utils.stats import RunningMeanStd


def test_running_mean_std():
    r = RunningMeanStd(shape=(1,))

    values = torch.arange(5, dtype=torch.float32).reshape(5, 1)
    r.update(values)
    values = torch.arange(5, 10, dtype=torch.float32).reshape(5, 1)
    r.update(values)
    values = torch.arange(10, 15, dtype=torch.float32).reshape(5, 1)
    r.update(values)
    values = torch.arange(15, 20, dtype=torch.float32).reshape(5, 1)
    r.update(values)
    assert r.mean == 9.5
    # Testing with parameter unbiased=False because the original numpy implementation uses it by default
    assert torch.allclose(r.std, torch.std(torch.arange(20, dtype=torch.float32), unbiased=False))
    assert torch.allclose(r.variance, torch.var(torch.arange(20, dtype=torch.float32), unbiased=False))
