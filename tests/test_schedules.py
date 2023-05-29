

from marl.utils import ExpSchedule, LinearSchedule


def test_exp_schedule():
    sched = ExpSchedule(16, 1, 5)
    expected = [16, 8, 4, 2, 1, 1, 1, 1]
    for exp in expected:
        assert sched == exp
        sched.update()

def test_linear_schedule():
    sched = LinearSchedule(10, 0, 10)
    expected = [10] + list(reversed(range(10))) + [0] * 10
    for exp in expected:
        assert sched == exp
        sched.update()