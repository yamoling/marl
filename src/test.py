from marl.utils import MultiSchedule, ConstantSchedule, LinearSchedule

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
