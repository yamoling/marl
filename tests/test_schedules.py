from marl.utils import ExpSchedule, LinearSchedule, ConstantSchedule, MultiSchedule


def test_exp_schedule():
    sched = ExpSchedule(16, 1, 5)
    expected = [16, 8, 4, 2, 1, 1, 1, 1]
    for exp in expected:
        assert sched == exp
        assert exp == sched
        sched.update()


def test_linear_schedule():
    sched = LinearSchedule(10, 0, 10)
    expected = list(range(10, 0, -1)) + [0, 0, 0, 0, 0, 0, 0, 0]
    for exp in expected:
        assert sched == exp
        sched.update()

    for i, exp in enumerate(expected):
        sched.update(i)
        assert sched == exp


def test_constant_schedule():
    sched = ConstantSchedule(10.0)
    for _ in range(20):
        assert sched == 10.0
        sched.update()


def test_multi_schedule():
    sched = MultiSchedule(
        {
            0: ConstantSchedule(10),
            5: LinearSchedule(10, 0, 10),
        }
    )

    for _ in range(5):
        assert sched.value == 10
        sched.update()
    for x in range(10, 5, -1):
        assert sched.value == x
        sched.update()

    sched.update(4)
    assert sched.value == 10

    sched.update(6)
    assert sched.value == 9
