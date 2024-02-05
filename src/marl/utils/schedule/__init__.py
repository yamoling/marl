from .schedule import Schedule, LinearSchedule, ExpSchedule, ConstantSchedule


from marl.utils.registry import make_registry

registry, from_summary = make_registry(Schedule, [schedule])
