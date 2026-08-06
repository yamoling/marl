from lle import Action

from marl.env import LLEConfig

MAP_PATH = "../lle/lift.toml"

N, S, E, W, STAY, TRIGGER = (
    Action.NORTH,
    Action.SOUTH,
    Action.EAST,
    Action.WEST,
    Action.STAY,
    Action.TRIGGER,
)
# Hand-solved action sequence for lift.toml, from lle's examples/main.py.
SOLUTION = [
    (E, E),
    (E, E),
    (E, STAY),
    (STAY, N),
    (STAY, TRIGGER),
    (W, E),
    (TRIGGER, STAY),
    (W, S),
    (W, STAY),
    (S, STAY),
]


def test_lift_button_map_has_lift_and_button():
    env = LLEConfig(MAP_PATH, obs_type="flattened").make()
    world = env.unwrapped.world
    assert len(world.lifts) == 1
    assert len(world.buttons) == 2


def test_lift_button_solution_reaches_terminal_state():
    """The button->lift mechanic must fire correctly through marl's wrapped env
    (agent-id extras, time limit, ...), not just the raw lle env."""
    env = LLEConfig(MAP_PATH, obs_type="flattened").make()
    env.reset()
    step = None
    for actions in SOLUTION:
        step = env.step([action.value for action in actions])
        if step.is_terminal:
            break
    assert step is not None and step.is_terminal


def test_lift_button_random_rollout_does_not_crash():
    env = LLEConfig(MAP_PATH, obs_type="flattened").make()
    env.reset()
    for _ in range(100):
        step = env.step(env.sample_action())
        if step.is_terminal:
            env.reset()
