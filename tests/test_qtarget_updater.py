"""Tests for marl.algos.qtarget_updater: HardUpdate and SoftUpdate."""

import pytest
import torch

from marl.algos.qtarget_updater import HardUpdate, SoftUpdate


def _params(*values: float):
    return [torch.nn.Parameter(torch.tensor([v])) for v in values]


class TestAddParameters:
    def test_appends_matching_pairs(self):
        updater = SoftUpdate(tau=0.5)
        params = _params(1.0, 2.0)
        targets = _params(0.0, 0.0)
        updater.add_parameters(params, targets)
        assert updater.parameters == params
        assert updater.target_parameters == targets

    def test_raises_on_shape_mismatch(self):
        updater = SoftUpdate(tau=0.5)
        param = torch.nn.Parameter(torch.zeros(3))
        target = torch.nn.Parameter(torch.zeros(4))
        with pytest.raises(AssertionError):
            updater.add_parameters([param], [target])


class TestSoftUpdate:
    def test_rejects_invalid_tau(self):
        with pytest.raises(AssertionError):
            SoftUpdate(tau=0.0)
        with pytest.raises(AssertionError):
            SoftUpdate(tau=1.0)

    def test_moves_target_towards_param_by_tau_fraction(self):
        updater = SoftUpdate(tau=0.1)
        param = torch.nn.Parameter(torch.tensor([1.0]))
        target = torch.nn.Parameter(torch.tensor([0.0]))
        updater.add_parameters([param], [target])
        updater.update(time_step=1)
        assert target.item() == pytest.approx(0.1)

    def test_repeated_updates_converge_target_to_param(self):
        updater = SoftUpdate(tau=0.5)
        param = torch.nn.Parameter(torch.tensor([2.0]))
        target = torch.nn.Parameter(torch.tensor([0.0]))
        updater.add_parameters([param], [target])
        for t in range(1, 30):
            updater.update(t)
        assert target.item() == pytest.approx(2.0, abs=1e-6)

    def test_update_returns_empty_logs(self):
        updater = SoftUpdate(tau=0.1)
        assert updater.update(0) == {}


class TestHardUpdate:
    def test_rejects_non_positive_period(self):
        with pytest.raises(AssertionError):
            HardUpdate(update_period=0)

    def test_target_unchanged_before_period_elapses(self):
        updater = HardUpdate(update_period=5)
        param = torch.nn.Parameter(torch.tensor([9.0]))
        target = torch.nn.Parameter(torch.tensor([0.0]))
        updater.add_parameters([param], [target])
        for _ in range(4):
            updater.update(0)
        assert target.item() == pytest.approx(0.0)

    def test_target_copies_param_exactly_at_period(self):
        updater = HardUpdate(update_period=5)
        param = torch.nn.Parameter(torch.tensor([9.0]))
        target = torch.nn.Parameter(torch.tensor([0.0]))
        updater.add_parameters([param], [target])
        for _ in range(5):
            updater.update(0)
        assert target.item() == pytest.approx(9.0)

    def test_target_stays_at_param_value_after_subsequent_updates_without_change(self):
        updater = HardUpdate(update_period=2)
        param = torch.nn.Parameter(torch.tensor([3.0]))
        target = torch.nn.Parameter(torch.tensor([0.0]))
        updater.add_parameters([param], [target])
        updater.update(0)
        updater.update(0)  # copy happens here (2nd call)
        assert target.item() == pytest.approx(3.0)
        param.data.fill_(7.0)
        updater.update(0)  # 3rd call, not a multiple of period yet
        assert target.item() == pytest.approx(3.0)
        updater.update(0)  # 4th call, copy happens
        assert target.item() == pytest.approx(7.0)
