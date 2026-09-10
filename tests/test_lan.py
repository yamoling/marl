"""LAN equations, recurrent execution, serialization, and device ownership."""

import math

import pytest
import torch
from marlenv import Episode, Transition
from marlenv.catalog import DiscreteMockEnv

from marl.algos import LAN, HardUpdate
from marl.env import EnvConfig
from marl.models.batch import EpisodeBatch


def setup_lan():
    config = EnvConfig.from_any(DiscreteMockEnv(n_agents=2, end_game=4), last_action=True)
    trainer = LAN.from_env(config, hidden_size=8, embedding_size=12, batch_size=2)
    return config.make(), trainer


def episode(env, trainer, truncate=False):
    agent = trainer.make_agent()
    agent.new_episode()
    obs, state = env.reset()
    result = Episode.new(obs, state)
    while not result.is_finished:
        action = agent.choose_action(obs).action
        step = env.step(action)
        if truncate and len(result) == 1:
            step.truncated = True
        elif not truncate and len(result) == 3:
            step.done = True
            step.truncated = False
        result.add(Transition.from_step(obs, state, action, step))
        obs, state = step.obs, step.state
    return result


def batch_for(env, trainer):
    return EpisodeBatch([episode(env, trainer), episode(env, trainer, True)]).for_individual_learners()


def test_proxy_and_double_q_targets_with_illegal_actions_and_time_limits():
    env, trainer = setup_lan()
    batch = batch_for(env, trainer)
    with torch.no_grad():
        for network in trainer.networks():
            for parameter in network.parameters():
                parameter.zero_()
        trainer.qnetwork.head.bias.copy_(torch.arange(env.n_actions))
        trainer.qtarget.head.bias.copy_(-torch.arange(env.n_actions))
        trainer.value_network.value[-1].bias.fill_(3)
        trainer.target_value.value[-1].bias.fill_(7)
    batch.next_available_actions[:] = False
    batch.next_available_actions[..., :2] = True
    _, values = trainer._compute_qvalues(batch)
    torch.testing.assert_close(values, batch.actions.float() + 3)
    targets = trainer._compute_qtargets(batch)
    # Online selects action 1; target would select 0. Illegal larger actions are excluded.
    torch.testing.assert_close(targets, batch.rewards + trainer.gamma * 6 * batch.not_dones)
    assert not targets.requires_grad
    torch.testing.assert_close(targets[3, 0], batch.rewards[3, 0])
    torch.testing.assert_close(targets[1, 1], batch.rewards[1, 1] + trainer.gamma * 6)


def test_recurrent_unroll_matches_execution_and_preserves_rollout_state():
    env, trainer = setup_lan()
    batch = batch_for(env, trainer)
    net = trainer.qnetwork
    net.reset_hidden_states()
    sequential = torch.cat(
        [net(batch.all_obs[t : t + 1], batch.all_extras[t : t + 1]) for t in range(len(batch.all_obs))]
    )
    saved = net._hidden_states.clone()
    unrolled = net.batch_qvalues(batch.all_obs, batch.all_extras, masks=batch.all_masks)
    torch.testing.assert_close(sequential, unrolled)
    torch.testing.assert_close(net._hidden_states, saved)
    # Changing one agent's observations cannot change the other agent's local outputs.
    changed = batch.all_obs.clone()
    changed[:, :, 0] += 10
    modified, _, _ = net.features(changed, batch.all_extras)
    torch.testing.assert_close(modified[:, :, 1], unrolled[:, :, 1])


def test_joint_gradients_masking_and_two_updates_per_episode():
    env, trainer = setup_lan()
    batch = batch_for(env, trainer)
    _, values = trainer._compute_qvalues(batch)
    targets = trainer._compute_qtargets(batch)
    loss, _ = trainer._compute_td_loss(values, targets, batch)
    loss.backward()
    assert trainer.qnetwork.gru.weight_ih_l0.grad.abs().sum() > 0
    assert trainer.value_network.embedding[0].weight.grad.abs().sum() > 0
    changed_targets = targets.clone()
    changed_targets[batch.masks == 0] += 1000
    changed_loss, _ = trainer._compute_td_loss(values, changed_targets, batch)
    torch.testing.assert_close(loss, changed_loss)
    first, second = episode(env, trainer), episode(env, trainer)
    assert trainer.update_episode(first, 0, 4) == {}
    logs = trainer.update_episode(second, 1, 8)
    assert math.isfinite(logs["td-loss"])
    assert trainer.target_updater._update_num == 2


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_device_transfer_optimizer_state_and_targets(device):
    env, trainer = setup_lan()
    batch = batch_for(env, trainer)
    trainer.train(0, batch)
    steps = [state["step"].item() for state in trainer.optimiser.state.values()]
    trainer.qnetwork.eval()  # Also move the saved training rollout state.
    trainer.to(torch.device(device))
    trainer.qnetwork.train()
    assert trainer.qnetwork._hidden_states.device.type == device
    for network in trainer.networks():
        assert all(p.device.type == device for p in network.parameters())
    online = list(trainer.qnetwork.parameters()) + list(trainer.value_network.parameters())
    assert {id(p) for p in online} == {id(p) for p in trainer.optimiser.param_groups[0]["params"]}
    assert steps == [state["step"].item() for state in trainer.optimiser.state.values()]
    logs = trainer.train(1, batch.to(torch.device(device)))
    assert math.isfinite(logs["td-loss"])
    updater = HardUpdate(1)
    updater.add_parameters(trainer.target_updater.parameters, trainer.target_updater.target_parameters)
    updater.update(1)
    for source, target in zip(updater.parameters, updater.target_parameters):
        torch.testing.assert_close(source, target)
    trainer.make_agent().new_episode()
    trainer.make_agent().choose_action(env.reset()[0])


def test_serialization_and_unconstrained_advantages():
    _, trainer = setup_lan()
    restored = LAN.from_json(trainer.to_json())
    assert isinstance(restored, LAN)
    assert restored.value_network.hidden_size == 8
    assert restored.updates_per_episode == 2
    with torch.no_grad():
        trainer.qnetwork.head.weight.zero_()
        trainer.qnetwork.head.bias.fill_(5)
    obs = torch.zeros(1, 2, trainer.qnetwork.obs_size)
    extras = torch.zeros(1, 2, trainer.qnetwork.extras_size)
    advantages, _, _ = trainer.qnetwork.features(obs, extras)
    torch.testing.assert_close(advantages, torch.full_like(advantages, 5))
    trainer.qnetwork.mean_center = True
    centered, _, _ = trainer.qnetwork.features(obs, extras)
    torch.testing.assert_close(centered, torch.zeros_like(centered))


def test_checkpoint_does_not_overwrite_online_weights_with_targets(tmp_path):
    _, trainer = setup_lan()
    with torch.no_grad():
        trainer.qnetwork.head.bias.fill_(1)
        trainer.qtarget.head.bias.fill_(2)
        trainer.value_network.value[-1].bias.fill_(3)
        trainer.target_value.value[-1].bias.fill_(4)
    trainer.save(tmp_path)
    _, restored = setup_lan()
    restored.load(tmp_path)
    for source, target in zip(trainer.networks(), restored.networks()):
        for key, value in source.state_dict().items():
            torch.testing.assert_close(value, target.state_dict()[key])


def test_exploration_anneals_before_replay_can_be_sampled():
    env, trainer = setup_lan()
    obs, state = env.reset()
    action = trainer.make_agent().choose_action(obs).action
    transition = Transition.from_step(obs, state, action, env.step(action))
    assert trainer.update_step(transition, 25_000)["epsilon"] == pytest.approx(0.525)
    assert trainer.update_step(transition, 50_000)["epsilon"] == pytest.approx(0.05)
