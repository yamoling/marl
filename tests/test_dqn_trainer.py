import marl
from copy import deepcopy
from marl.training.qtarget_updater import SoftUpdate, HardUpdate
from .utils import MockEnv, generate_episode, parameters_equal


def test_trainer_nodes_vdn():
    env = MockEnv(4)
    trainer = marl.training.DQNNodeTrainer(
        qnetwork=marl.nn.model_bank.MLP.from_env(env),
        memory=marl.models.TransitionMemory(50_000),
        train_policy=marl.policy.EpsilonGreedy.linear(1, 0.01, n_steps=20_000),
        mixer=marl.qlearning.VDN(env.n_agents),
    )

    assert len(trainer.updatables) == len(["epsilon", "loss", "target updater"])
    assert isinstance(trainer.target_params_updater, SoftUpdate)
    assert trainer.mixer is not None
    assert trainer.ir_module is None


def test_target_network_is_updated():
    env = MockEnv(4)
    batch_size = 4
    target_updater = HardUpdate(5)
    qnetwork = marl.nn.model_bank.MLP.from_env(env)
    trainer = marl.training.DQNNodeTrainer(
        qnetwork=qnetwork,
        memory=marl.models.TransitionMemory(50_000),
        train_policy=marl.policy.EpsilonGreedy.linear(1, 0.01, n_steps=20_000),
        target_updater=target_updater,
        train_interval=(1, "step"),
        batch_size=batch_size,
    )
    # Randomize the parameters
    trainer.qnetwork.randomize()
    prev_qnetwork_params = deepcopy(target_updater.parameters)
    prev_qtarget_params = deepcopy(target_updater.target_params)
    episode = generate_episode(env)
    assert len(episode) > batch_size + target_updater.update_period
    for time_step, transition in enumerate(episode):  # type: ignore
        trainer.update_step(transition, time_step)
        if time_step >= batch_size - 1:
            # Check that the update indeed happened to the qnetwork
            assert not parameters_equal(prev_qnetwork_params, list(qnetwork.parameters()))
            prev_qnetwork_params = deepcopy(list(qnetwork.parameters()))
            if target_updater.update_num % target_updater.update_period == 0:
                # If we should update the qtarget, check that it has indeed been updated and
                # that they are equal to the qnetwork weights
                assert not parameters_equal(prev_qtarget_params, target_updater.target_params)
                prev_qtarget_params = deepcopy(target_updater.target_params)
                assert parameters_equal(prev_qnetwork_params, prev_qtarget_params)
            else:
                # Otherwise, check that the qtarget has not been updated
                assert parameters_equal(prev_qtarget_params, target_updater.target_params)
