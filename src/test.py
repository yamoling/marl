import os
import sys
import time
import termios
import tty
import orjson

from lle import LLE, Action


def getkey():
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        while True:
            b = os.read(sys.stdin.fileno(), 3).decode()
            if len(b) == 3:
                k = ord(b[2])
            else:
                k = ord(b)
            key_mapping = {
                127: "backspace",
                10: "return",
                32: "space",
                9: "tab",
                27: "esc",
                65: "up",
                66: "down",
                67: "right",
                68: "left",
            }
            return key_mapping.get(k, chr(k))
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def get_action(agent_id: int, available_actions: list[Action]) -> Action:
    action = None
    while action is None:
        print(f"Enter an action for agent {agent_id} among {available_actions}: ", end="", flush=True)
        key = getkey()
        print(key)
        if key == "up":
            action = Action.NORTH
        elif key == "down":
            action = Action.SOUTH
        elif key == "left":
            action = Action.WEST
        elif key == "right":
            action = Action.EAST
        elif key == "space":
            action = Action.STAY
        else:
            print(f"Unrecognized key: {key}")
        if action not in available_actions:
            print(f"Action {action} not in available actions {available_actions}")
            action = None
    return action


if __name__ == "__main__":
    env = LLE.level(6).obs_type("layered").state_type("state").build()
    world = env._world
    world.reset()
    env.reset()
    done = False
    all_actions = []
    while not done:
        env.render()
        time.sleep(0.1)
        env.render()
        actions = list[int]()
        for agent, available in enumerate(world.available_actions()):
            actions.append(get_action(agent, available).value)
        all_actions.append(actions)
        step = env.step(actions)
        with open("data/actions.json", "wb") as f:
            f.write(orjson.dumps(all_actions))
        done = step.is_terminal
    env.render()
