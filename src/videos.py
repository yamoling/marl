import os

import orjson
from lle import LLE, World, WorldState
from marlenv import Builder, MARLEnv
from marlenv.wrappers import VideoRecorder

import marl


def replay(rundir: str, env, t: int):
    images, r = [], 0
    env.reset()
    images.append(env.get_image())
    actions_file = os.path.join(rundir, "test", str(t), "0", "actions.json")
    with open(actions_file) as f:
        actions = orjson.loads(f.read())
    for action in actions:
        s = env.step(action)
        images.append(env.get_image())
        r += s.reward
    print(r)
    return images


def with_shaping():
    gamma = 0.95

    env = LLE.level(6).obs_type("layered").state_type("layered").build()

    width = env.width
    height = env.height

    env = Builder(env).time_limit(width * height // 2).agent_id().build()

    qnetwork = marl.nn.model_bank.CNN.from_env(env)
    dqn = marl.agents.DQN(
        qnetwork=qnetwork,
        train_policy=marl.policy.ArgMax(),
        test_policy=marl.policy.ArgMax(),
    )
    dqn.set_testing()


def swap_initial_pos(exp: marl.Experiment):
    exp.env.reset()
    world: World = exp.env.wrapped.wrapped.wrapped.world  # type: ignore
    state = world.get_state()
    agents_positions = world.agents_positions
    new_agents_pos = [agents_positions[1], agents_positions[0], agents_positions[2], agents_positions[3]]
    new_state = WorldState(new_agents_pos, state.gems_collected, state.agents_alive)
    world.set_state(new_state)


def run(env: MARLEnv, agent: marl.Agent, checkpoint: str):
    agent.load(checkpoint)
    obs = env.get_observation()
    # env.render()
    is_terminal = False
    while not is_terminal:
        action = agent.choose_action(obs)
        step = env.step(action)
        # env.render()
        obs = step.obs
        is_terminal = step.is_terminal


if __name__ == "__main__":
    ------
    # The problem is that the previous experiment was launched with a previous version of multi-agent-rlenv
    # So the attributes do not match anymore.
    # A new experiment should be started, making sure that the reward scheme is correct and the subgoals are indeed given.
    exp = marl.Experiment.load("logs/LLE-lvl6-PBRS-VDN")
    exp.env = VideoRecorder(exp.env, end_pause_frames=5, initial_pause_frames=5, fps=3)
    swap_initial_pos(exp)
    run(exp.env, exp.agent, "logs/LLE-lvl6-PBRS-VDN/run_2025-02-26_12:59:23.108388_seed=0/test/650000")
