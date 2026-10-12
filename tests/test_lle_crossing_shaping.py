import numpy as np
from lle import Action

from marl.env import LLEConfig

S, X = Action.SOUTH.value, Action.STAY.value


def _rollout(env, actions):
    env.reset()
    rewards = []
    for joint_action in actions:
        step = env.step(np.array(joint_action))
        rewards.append(float(step.reward[0]))
        if step.is_terminal:
            break
    return rewards, step


def test_far_sides_of_level_6():
    """The far side of the second laser excludes the dead end behind it. @ai-generated"""
    shaping = LLEConfig(6, pbrs="joint", pbrs_subgoals="crossing").make().unwrapped.reward_strategy
    assert shaping.subgoals[1] == {(5, j) for j in range(1, 7)}
    assert shaping.subgoals[3] == {(7, j) for j in range(7)}


def test_crossing_rewards_entering_and_crossing():
    """@ai-generated"""
    env = LLEConfig(6, pbrs="joint", pbrs_subgoals="crossing").make()
    rewards, step = _rollout(env, [[S, X, X, X]] * 5)
    assert rewards == [0.0, 0.0, 0.0, 0.5, 0.5]
    assert step.obs.extras[0][:4].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_dying_in_a_laser_is_not_rewarded():
    """@ai-generated"""
    env = LLEConfig(6, pbrs="joint", pbrs_subgoals="crossing").make()
    rewards, step = _rollout(env, [[X, S, X, X]] * 4)
    assert step.done
    assert rewards[-1] == -1.0
    assert step.obs.extras[0][4:8].tolist() == [0.0] * 4


def test_reset_clears_subgoals():
    """@ai-generated"""
    env = LLEConfig(6, pbrs="individual", pbrs_subgoals="crossing").make()
    _rollout(env, [[S, X, X, X]] * 5)
    obs, _ = env.reset()
    assert env.extras_shape[0] == 4 + 1 + 4  # subgoals, time ratio, agent id
    assert not obs.extras[:, :4].any()


def test_legacy_configs_are_unchanged():
    """Legacy `pbrs=False` configurations do not get shaped. @ai-generated"""
    assert LLEConfig(6, pbrs=False).name == "LLE-lvl6"  # type: ignore[arg-type]
    assert LLEConfig(6, pbrs="joint").name == "LLE-lvl6-PBRS"
