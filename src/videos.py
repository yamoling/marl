import os

import orjson
from lle import LLE, World, WorldState
from marlenv import Builder, MARLEnv
from marlenv.wrappers import VideoRecorder

import marl


def swap_initial_pos(exp: marl.Experiment):
    exp.env.reset()
    world: World = exp.env.wrapped.wrapped.wrapped.world  # type: ignore
    state = world.get_state()
    agents_positions = world.agents_positions
    new_agents_pos = [agents_positions[1], agents_positions[0], agents_positions[2], agents_positions[3]]
    new_state = WorldState(new_agents_pos, state.gems_collected, state.agents_alive)
    world.set_state(new_state)


def swap_laser_colours(exp: marl.Experiment):
    exp.env.reset()
    world: World = exp.env.wrapped.wrapped.wrapped.world  # type: ignore
    for source in world.laser_sources:
        if source.pos == (4, 0):
            source.set_colour(1)
        elif source.pos == (6, 12):
            source.set_colour(0)


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
    # The problem is that the previous experiment was launched with a previous version of multi-agent-rlenv
    # So the attributes do not match anymore.
    # A new experiment should be started, making sure that the reward scheme is correct and the subgoals are indeed given.
    exp = marl.Experiment.load("logs/debug")
    exp.env = VideoRecorder(exp.env, end_pause_frames=5, initial_pause_frames=5, fps=3)
    # swap_initial_pos(exp)
    swap_laser_colours(exp)
    run(exp.env, exp.agent, "logs/debug/run_2025-03-02_10:26:38.084753_seed=0/test/0")
