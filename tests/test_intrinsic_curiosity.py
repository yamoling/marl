import marl
import numpy as np

from rlenv import Transition
from lle import LLE, Action, ObservationType
from marl.models.batch import TransitionBatch
from marl.models import NN
from marl.intrinsic_reward import RandomNetworkDistillation


def _test_rnd_no_reward_normalisation(env: LLE, target: NN):
    rnd = RandomNetworkDistillation(target, normalise_rewards=False, reward_size=env.reward_size)
    transitions = []
    obs = env.reset()
    for _ in range(64):
        action = np.array([Action.STAY.value, Action.STAY.value], dtype=np.int32)
        obs_, r, done, truncated, info = env.step(action)
        transitions.append(Transition(obs, action, r, done, info, obs_, truncated))
        obs = obs_
    train_batch = TransitionBatch(transitions)

    transitions = []
    actions = (
        [[Action.SOUTH.value, Action.SOUTH.value]] * 5
        + [[Action.EAST.value, Action.STAY.value]] * 5
        + [[Action.STAY.value, Action.EAST.value]] * 5
        + [[Action.NORTH.value, Action.NORTH.value]] * 5
        + [[Action.STAY.value, Action.WEST.value]] * 5
        + [[Action.WEST.value, Action.STAY.value]] * 5
    )
    for action in actions:
        action = np.array(action, dtype=np.int32)
        obs_, r, done, truncated, info = env.step(action)
        transitions.append(Transition(obs, action, r, done, info, obs_, truncated))
        obs = obs_
    test_batch = TransitionBatch(transitions)

    irs = []
    # Train RND
    for _ in range(2000):
        ir = rnd.compute(train_batch).mean().item()
        irs.append(ir)
        assert ir >= 0
        rnd.update(_)
    initial_ir = sum(irs[0:100]) / 100
    ir_train_set = rnd.compute(train_batch).mean().item()
    ir_test_set = rnd.compute(test_batch).mean().item()
    assert ir_train_set < initial_ir
    assert ir_train_set < ir_test_set


def test_rnd_linear():
    env = LLE.level(2)
    target = marl.nn.model_bank.MLP(
        input_size=env.observation_shape[0],
        extras_size=env.extra_feature_shape[0],
        hidden_sizes=(64, 64, 64),
        output_shape=(512,),
    )
    _test_rnd_no_reward_normalisation(env, target)


def test_rnd_conv():
    env = LLE.level(2, ObservationType.LAYERED)
    target = marl.nn.model_bank.CNN(env.observation_shape, env.extra_feature_shape[0], (512,))
    _test_rnd_no_reward_normalisation(env, target)
