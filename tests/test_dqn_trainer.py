import marl
from copy import deepcopy
from marl.training.dqn_trainer import SoftUpdate, HardUpdate
from .utils import MockEnv, generate_episode, parameters_equal


def test_trainer_nodes_vdn():
    env = MockEnv(4)
    trainer = marl.training.DQNTrainer(
        qnetwork=marl.nn.model_bank.MLP256.from_env(env),
        train_policy=marl.policy.EpsilonGreedy.linear(1, 0.01, n_steps=20_000),
        mixer=marl.qlearning.VDN(env.n_agents),
    )
    trainer.show()

    assert len(trainer.root.children) == len(["next qvalues", "next qvalues mixer", "qvalues", "qvalues mixer", "targets", "loss"])
    assert len(trainer.loss.parents) == len(["qvalues", "qtargets", "batch"])
    assert isinstance(trainer.target_params_updater, SoftUpdate)
    assert trainer.mixer is not None
    assert trainer.ir_module is None


def test_target_network_is_updated():
    env = MockEnv(4)
    batch_size = 4
    target_update = 5
    trainer = marl.training.DQNTrainer(
        qnetwork=marl.nn.model_bank.MLP.from_env(env),
        train_policy=marl.policy.EpsilonGreedy.linear(1, 0.01, n_steps=20_000),
        target_update=HardUpdate(target_update),
        train_every="step",
        update_interval=1,
        batch_size=batch_size,
    )
    qnetwork_params = deepcopy(list(trainer.qnetwork.parameters()))
    qtarget_params = deepcopy(list(trainer.qtarget.parameters()))
    # Randomize the parameters
    trainer.qnetwork.randomize()
    episode = generate_episode(env)
    assert len(episode) > batch_size + target_update
    for time_step, transition in enumerate(episode):  # type: ignore
        trainer.update_step(transition, time_step)
        if time_step >= batch_size - 1:
            # Check that the update indeed happened to the qnetwork
            assert not parameters_equal(qnetwork_params, list(trainer.qnetwork.parameters()))
            qnetwork_params = deepcopy(list(trainer.qnetwork.parameters()))
            if time_step % target_update == 0:
                # If we should update the qtarget, check that it has indeed been updated and
                # that they are equal to the qnetwork weights
                assert not parameters_equal(qtarget_params, trainer.qtarget.parameters())
                qtarget_params = deepcopy(list(trainer.qtarget.parameters()))
                assert parameters_equal(qnetwork_params, qtarget_params)
            else:
                # Otherwise, check that the qtarget has not been updated
                assert parameters_equal(qtarget_params, trainer.qtarget.parameters())
