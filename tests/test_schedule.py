"""
Tests for marl.utils.schedule.

Covers LinearSchedule, ExpSchedule, ConstantSchedule and RoundedSchedule, as well as
the operator-overloading mixins defined on the abstract `Schedule` base class.
"""

import math

import pytest

from marl.utils.schedule import ConstantSchedule, ExpSchedule, LinearSchedule, RoundedSchedule, Schedule


class TestLinearSchedule:
    def test_starts_at_start_value(self):
        s = Schedule.linear(1.0, 0.0, 10)
        assert s.value == pytest.approx(1.0)

    def test_reaches_end_value_exactly_at_n_steps(self):
        s = Schedule.linear(1.0, 0.0, 10)
        for _ in range(10):
            s.update()
        assert s.value == pytest.approx(0.0)

    def test_clamps_to_end_value_past_n_steps(self):
        s = Schedule.linear(1.0, 0.0, 10)
        for _ in range(50):
            s.update()
        assert s.value == pytest.approx(0.0)

    def test_intermediate_value_is_linear(self):
        s = Schedule.linear(0.0, 10.0, 10)
        s.update(step=5)
        assert s.value == pytest.approx(5.0)

    def test_forcing_a_step_sets_absolute_position(self):
        s = Schedule.linear(0.0, 100.0, 100)
        s.update(step=50)
        assert s.value == pytest.approx(50.0)
        s.update(step=10)
        assert s.value == pytest.approx(10.0)

    def test_increasing_schedule(self):
        s = Schedule.linear(0.0, 1.0, 4)
        values = []
        for _ in range(4):
            s.update()
            values.append(s.value)
        assert values == pytest.approx([0.25, 0.5, 0.75, 1.0])


class TestExpSchedule:
    def test_starts_at_start_value(self):
        s = Schedule.exp(1.0, 0.01, 10)
        assert s.value == pytest.approx(1.0)

    def test_reaches_end_value_past_n_steps(self):
        s = Schedule.exp(1.0, 0.01, 10)
        for _ in range(50):
            s.update()
        assert s.value == pytest.approx(0.01)

    def test_monotonically_decreasing(self):
        s = Schedule.exp(1.0, 0.01, 20)
        previous = s.value
        for _ in range(19):
            s.update()
            assert s.value <= previous
            previous = s.value

    def test_at_last_step_before_clamp_matches_end_value(self):
        n_steps = 10
        s = Schedule.exp(1.0, 0.01, n_steps)
        s.update(step=n_steps - 1)
        assert s.value == pytest.approx(0.01, rel=1e-6)


class TestConstantSchedule:
    def test_value_is_start_value(self):
        s = Schedule.constant(0.5)
        assert s.value == pytest.approx(0.5)

    def test_update_does_not_change_value(self):
        s = Schedule.constant(0.5)
        for _ in range(100):
            s.update()
        assert s.value == pytest.approx(0.5)

    def test_end_value_equals_start_value(self):
        s = Schedule.constant(0.3)
        assert s.end_value == pytest.approx(0.3)


class TestRoundedSchedule:
    def test_rounds_the_wrapped_value(self):
        inner = Schedule.linear(0.0, 1.0, 3)
        rounded = inner.rounded(n_digits=1)
        inner.update(step=1)
        # inner.value = 1/3 = 0.333... but `rounded` computes from `inner.value` directly.
        assert rounded.value == pytest.approx(round(inner.value, 1))

    def test_update_delegates_to_wrapped_schedule(self):
        inner = Schedule.linear(0.0, 10.0, 10)
        rounded = inner.rounded(n_digits=0)
        rounded.update(step=5)
        assert inner.value == pytest.approx(5.0)
        assert rounded.value == pytest.approx(5.0)

    def test_name_is_prefixed(self):
        inner = Schedule.linear(0.0, 1.0, 10)
        rounded = RoundedSchedule(inner, n_digits=2)
        assert rounded.name == "RoundedLinearSchedule"


class TestOperatorOverloading:
    def test_multiplication(self):
        s = Schedule.constant(2.0)
        assert s * 3 == pytest.approx(6.0)
        assert 3 * s == pytest.approx(6.0)

    def test_addition(self):
        s = Schedule.constant(2.0)
        assert s + 1 == pytest.approx(3.0)
        assert 1 + s == pytest.approx(3.0)

    def test_subtraction(self):
        s = Schedule.constant(2.0)
        assert s - 1 == pytest.approx(1.0)
        assert 5 - s == pytest.approx(3.0)

    def test_true_division(self):
        s = Schedule.constant(4.0)
        assert s / 2 == pytest.approx(2.0)
        assert 8 / s == pytest.approx(2.0)

    def test_power(self):
        s = Schedule.constant(2.0)
        assert s**3 == pytest.approx(8.0)

    def test_float_and_int_conversion(self):
        s = Schedule.constant(3.7)
        assert float(s) == pytest.approx(3.7)
        assert int(s) == 3

    def test_comparisons(self):
        s = Schedule.constant(5.0)
        assert s < 6
        assert s <= 5
        assert s > 4
        assert s >= 5

    def test_equality_between_two_schedules_compares_definition(self):
        a = Schedule.linear(0.0, 1.0, 10)
        b = Schedule.linear(0.0, 1.0, 10)
        assert a == b

    def test_equality_between_schedules_of_different_type_is_false(self):
        a = LinearSchedule(start_value=0.0, end_value=1.0, n_steps=10)
        b = ExpSchedule(start_value=0.1, end_value=1.0, n_steps=10)
        assert not (a == b)

    def test_equality_with_plain_number_compares_current_value(self):
        s = Schedule.constant(5.0)
        assert s == 5.0
        assert s != 4.0

    def test_neg_and_pos(self):
        s = Schedule.constant(3.0)
        assert -s == pytest.approx(-3.0)
        assert +s == pytest.approx(3.0)
