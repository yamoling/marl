import marl
import json
import rlenv
import laser_env
import numpy as np
import time
import shutil

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
        
        experiment = marl.Experiment.create(logdir, vdn, env)
        runner = experiment.create_runner("csv")
        runner.train(n_steps=200, test_interval=50, n_tests=1)

        def check(time_step: int):
            episode_dir = f"{runner.rundir}/test/{time_step}/0"
            restored = experiment.replay_episode(episode_dir)
            with (open(episode_dir + "/actions.json", "r") as f,
                open(episode_dir + "/metrics.json", "r") as m):
                logged_actions = np.array(json.load(f))
                logged_metrics = json.load(m)
            assert np.array_equal(restored.episode.actions, logged_actions)
            assert restored.episode.metrics == logged_metrics
        for time_step in range(0, 201, 50):
            check(time_step)
    finally:
        shutil.rmtree(logdir)

    
