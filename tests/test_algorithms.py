import marl
import json
import rlenv
import laser_env
import numpy as np
import time
import shutil


def test_algo_from_summary():
    # TODO
    assert False


def test_save_replay_episode():
    logdir = f"logs/test-{time.time()}"
    try:
        # Train for a given amount of steps and save the weights
        # Then, store the weights and check that the actions performed are the same
        env = rlenv.Builder(laser_env.StaticLaserEnv("lvl3")).time_limit(30).build()
        vdn = (marl.DeepQBuilder()
            .qnetwork_default(env)
            # Small batch size for testing
            .batch_size(8)
            # Very big learning rate to see the difference when testing
            .optimizer("adam", lr=0.1)
            .build())
        
        experiment = marl.Experiment.create(logdir, vdn, env, 200, 5)
        runner = experiment.create_runner("csv")
        runner.train(n_tests=1)

        def check(time_step: int):
            episode_dir = f"{runner.rundir}/test/{time_step}/0"
            restored = experiment.replay_episode(episode_dir)
            with open(episode_dir + "/actions.json", "r") as f:
                logged_actions = np.array(json.load(f))
            assert np.array_equal(restored.episode.actions, logged_actions)
        for time_step in range(0, 201, 50):
            check(time_step)
    finally:
        shutil.rmtree(logdir)

    
