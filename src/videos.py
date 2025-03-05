from lle import World, WorldState
from marlenv.wrappers import VideoRecorder
import cv2
import os
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
    env._recorder = cv2.VideoWriter("swapped-pos", cv2.VideoWriter_fourcc(*"mp4v"), env.fps, (width, height))  # type: ignore
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


def get_checkpoint(logdir: str, run_num: int, step: int):
    runs = [run for run in os.listdir(logdir) if run.endswith(f"seed={run_num}")]
    return os.path.join(logdir, runs[0], "test", str(step))


def run(exp: marl.Experiment, run_num: int, test_step: int):
    checkpoint = get_checkpoint(exp.logdir, run_num, test_step)
    agent = exp.agent
    env = exp.env
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
    TEST_STEP = 980_000
    RUN_NUM = 0
    exp = marl.Experiment.load("logs/pbrs-randomized_lasers")
    exp.env = VideoRecorder(exp.env, end_pause_frames=5, initial_pause_frames=5, fps=3)
    runs = sorted(list(exp.runs), key=lambda r: r.seed)
    the_run = runs[RUN_NUM]
    actions = the_run.get_test_actions(TEST_STEP, 0)
    seed = marl.Runner.get_test_seed(TEST_STEP, 0)
    episode = exp.env.replay(actions, seed)
    print(len(episode))
    # swap_initial_pos(exp)
    # swap_laser_colours(exp)
    # run(exp, 8, 975000)
