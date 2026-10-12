import runpy
from pathlib import Path

import numpy as np
import optuna

import marl


def test_softmax_policy_defaults_to_environment_actions():
    """Preserve the primitive-action search space for other algorithms. @ai-generated"""
    tuning = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "tuning.py"))
    env = marl.env.LLEConfig(6)
    trial = optuna.trial.FixedTrial({"train_policy.__type__": "softmax", "train_policy.tau": 1.0})
    policy, noisy = tuning["suggest_train_policy"](trial, env, 10_000)

    actions = policy.get_action(np.zeros((env.n_agents, env.n_actions), dtype=np.float32))

    assert not noisy
    assert policy.n_actions == env.n_actions
    assert actions.shape == (env.n_agents,)
    assert np.all((actions >= 0) & (actions < env.n_actions))


def test_haven_macro_softmax_uses_subgoal_actions():
    """Exercise macro action selection when subgoal and primitive counts differ. @ai-generated"""
    tuning = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "tuning.py"))
    n_subgoals = tuning["HAVEN_N_SUBGOALS"]
    env = marl.env.LLEConfig(6, obs_type="perspective", state_type="flattened", extra_padding_size=n_subgoals)
    trial = optuna.trial.FixedTrial(
        {
            "k": 5,
            "qnetwork.duelling": False,
            "memory_size": 5_000,
            "lr": 0.0003,
            "batch_size": 32,
            "optimiser_type": "adam",
            "train_policy.__type__": "softmax",
            "train_policy.tau": 1.0,
            "target_updater.__type__": "soft",
            "target_updater.tau": 0.01,
            "worker.epsilon_end": 0.05,
            "worker.epsilon_n_steps": 1_000,
            "meta.grad_norm_clipping": 10.0,
            "worker.grad_norm_clipping": 10.0,
        }
    )
    trainer = tuning["make_haven_trainer"](trial, env, 10_000, {"n_agents": env.n_agents, "n_actions": env.n_actions, "gamma": 0.99})
    policy = trainer.meta_trainer.train_policy

    assert env.n_actions != n_subgoals
    assert policy.n_actions == n_subgoals
    actions = policy.get_action(np.zeros((env.n_agents, n_subgoals), dtype=np.float32))
    assert actions.shape == (env.n_agents,)
    assert np.all((actions >= 0) & (actions < n_subgoals))
