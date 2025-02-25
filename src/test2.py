from lle import LLE, Action
import time
import marlenv


def create_lle():
    n_steps = 1_000_000
    test_interval = 5000
    gamma = 0.95
    env = LLE.level(6).obs_type("layered").state_type("state").pbrs(gamma).build()
    env = marlenv.Builder(env).agent_id().time_limit(78, add_extra=True).build()

    actions = [
        [Action.SOUTH.value] * 4,
        [Action.SOUTH.value] * 4,
        [Action.SOUTH.value] * 4,
        [Action.SOUTH.value] * 2 + [Action.STAY.value] * 2,
    ]

    for a in actions:
        s = env.step(a)
        print(s.reward)
        env.render()
        time.sleep(1)


create_lle()
