from .schedule import Schedule, LinearSchedule, ExpSchedule, ConstantSchedule


from marl.utils.registry import make_registry

register, from_dict = make_registry(Schedule, [schedule])

