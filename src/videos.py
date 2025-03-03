from lle import World, WorldState
from marlenv import MARLEnv
from marlenv.wrappers import VideoRecorder
import cv2

import marl


def swap_initial_pos(env: VideoRecorder):
    env.reset()
    world: World = env.wrapped.wrapped.wrapped.world  # type: ignore
    state = world.get_state()
    agents_positions = world.agents_positions
    new_agents_pos = [agents_positions[1], agents_positions[0], agents_positions[2], agents_positions[3]]
    new_state = WorldState(new_agents_pos, state.gems_collected, state.agents_alive)
    world.set_state(new_state)
    assert env._recorder is not None
    env._recorder.release()
    img = env.get_image()
    height, width, _ = img.shape
    env._recorder = cv2.VideoWriter("swapped-pos", cv2.VideoWriter_fourcc(*"mp4v"), env.fps, (width, height))
    for _ in range(env.initial_pause_frames):
        env._recorder.write(img)
    env._recorder.write(img)


def swap_laser_colours(exp: marl.Experiment):
    world: World = exp.env.wrapped.wrapped.wrapped.world  # type: ignore
    for source in world.laser_sources:
        if source.pos == (4, 0):
            source.set_colour(1)
        elif source.pos == (6, 12):
            source.set_colour(0)
    exp.env.reset()


def run(env: MARLEnv, agent: marl.Agent, checkpoint: str):
    agent.load(checkpoint)
    obs = env.get_observation()
    is_terminal = False
    while not is_terminal:
        action = agent.choose_action(obs)
        step = env.step(action)
        obs = step.obs
        is_terminal = step.is_terminal


if __name__ == "__main__":
    # The problem is that the previous experiment was launched with a previous version of multi-agent-rlenv
    # So the attributes do not match anymore.
    # A new experiment should be started, making sure that the reward scheme is correct and the subgoals are indeed given.
    exp = marl.Experiment.load("logs/shaped-lle-reward1-two-sources")
    exp.env = VideoRecorder(exp.env, end_pause_frames=5, initial_pause_frames=5, fps=3)
    exp.env.reset()
    # swap_initial_pos(exp)
    swap_laser_colours(exp)
    run(exp.env, exp.agent, "logs/shaped-lle-reward1-two-sources/run_2025-03-02_17:29:46.117007_seed=0/test/1000000")
