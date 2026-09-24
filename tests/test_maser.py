import math
from copy import deepcopy
from typing import Any

import pytest
import torch
from marlenv import Episode, Transition

from marl import algos
from marl.algos.maser import MASER, at_timestep
from marl.env import LLEConfig
from marl.models import EpisodeMemory, Trainer, TransitionMemory
from marl.models.batch import EpisodeBatch, TransitionBatch
from marl.nn import mixers
from marl.nn.model_bank import qnetworks


def env_config(time_limit: int = 15):
    return LLEConfig(6, obs_type="flattened", state_type="flattened", time_limit=time_limit)


def make_trainer(recurrent: bool = False, transition_memory: bool = False, **kwargs: Any) -> MASER:
    env = env_config()
    if recurrent:
        qnetwork = qnetworks.QRNN.from_env(env, mlp_head_sizes=(16,), mlp_tail_sizes=(16,))
    else:
        qnetwork = qnetworks.QMLP.from_env(env, hidden_sizes=(16,))
    kwargs.setdefault("batch_size", 4)
    kwargs.setdefault("representation_hidden_size", 16)
    kwargs.setdefault("mixer", mixers.QMix.from_env(env, embed_size=8, hypernet_embed_size=8))
    memory = TransitionMemory(64) if transition_memory else EpisodeMemory(64)
    return algos.MASER(qnetwork, memory, train_interval=(1, "step" if transition_memory else "episode"), **kwargs)


def collect_episode(trainer: MASER, time_limit: int) -> Episode:
    env = env_config(time_limit).make()
    agent = trainer.make_agent()
    obs, state = env.reset()
    episode = Episode.new(obs, state)
    done = False
    while not done:
        action = agent.choose_action(obs)
        step = env.step(action.action)
        episode.add(Transition.from_step(obs, state, action, step))
        obs, state = step.obs, step.state
        done = step.done or step.truncated
    return episode


def padded_batch(trainer: MASER):
    """Episodes of different lengths so that the batch contains padding."""
    return EpisodeBatch([collect_episode(trainer, time_limit) for time_limit in (3, 5, 8)])


def test_at_timestep_selects_one_item_per_episode_and_agent():
    tensor = torch.arange(4 * 2 * 3 * 2).reshape(4, 2, 3, 2)
    timesteps = torch.tensor([[0, 3, 1], [2, 2, 0]])
    selected = at_timestep(tensor, timesteps)
    assert selected.shape == (1, 2, 3, 2)
    for b in range(2):
        for n in range(3):
            torch.testing.assert_close(selected[0, b, n], tensor[timesteps[b, n], b, n])


def test_subgoals_follow_individual_values_when_alpha_is_one():
    trainer = make_trainer(alpha=1.0)
    greedy = torch.tensor([[[5.0, 0.0]], [[1.0, 2.0]], [[0.0, 9.0]]])  # (time=3, batch=1, agents=2)
    qtotal = torch.tensor([[0.0], [100.0], [0.0]])
    padding = torch.zeros(3, 1, dtype=torch.bool)
    assert trainer._select_subgoals(greedy, qtotal, padding).tolist() == [[0, 2]]


def test_subgoals_are_shared_when_alpha_is_zero():
    trainer = make_trainer(alpha=0.0)
    greedy = torch.tensor([[[5.0, 0.0]], [[1.0, 2.0]], [[0.0, 9.0]]])
    qtotal = torch.tensor([[0.0], [100.0], [0.0]])
    padding = torch.zeros(3, 1, dtype=torch.bool)
    assert trainer._select_subgoals(greedy, qtotal, padding).tolist() == [[1, 1]]


def test_subgoals_mix_individual_and_total_values():
    trainer = make_trainer(alpha=0.5)
    # Scores: agent 0 -> [2.5, 1.5, 3.0], agent 1 -> [0.0, 1.5, 4.0 + 0.5]
    greedy = torch.tensor([[[5.0, 0.0]], [[1.0, 1.0]], [[4.0, 8.0]]])
    qtotal = torch.tensor([[0.0], [4.0], [4.0]])
    padding = torch.zeros(3, 1, dtype=torch.bool)
    assert trainer._select_subgoals(greedy, qtotal, padding).tolist() == [[2, 2]]


def test_subgoals_are_never_selected_in_padding():
    trainer = make_trainer(alpha=0.5)
    greedy = torch.tensor([[[1.0, 1.0]], [[2.0, 2.0]], [[99.0, 99.0]]])
    qtotal = torch.tensor([[0.0], [0.0], [99.0]])
    padding = torch.tensor([[False], [False], [True]])
    assert trainer._select_subgoals(greedy, qtotal, padding).tolist() == [[1, 1]]


def test_reward_design_matches_equations_3_and_4():
    trainer = make_trainer(intrinsic_weight=0.5)
    rewards = torch.tensor([[1.0]])
    greedy = torch.tensor([[[0.0, math.log(3.0)]]])  # softmax -> [0.25, 0.75]
    intrinsic = torch.tensor([[[-2.0, -4.0]]])
    proxy, individual = trainer._design_rewards(rewards, greedy, intrinsic)
    # R = 1 + 0.5 * mean(-2, -4) = -0.5
    torch.testing.assert_close(proxy, torch.tensor([[-0.5]]))
    # r^i = softmax_i * R + 0.5 * r^i_int
    torch.testing.assert_close(individual, torch.tensor([[[0.25 * -0.5 - 1.0, 0.75 * -0.5 - 2.0]]]))


def test_episodic_correction_is_kl_to_uniform_over_available_actions():
    qvalues = torch.tensor([[1.0, 1.0, 50.0], [0.0, 2.0, 7.0]], requires_grad=True)
    available = torch.tensor([[True, True, False], [True, True, True]])
    kl = MASER._uniform_kl(qvalues, available)
    probs = torch.softmax(torch.tensor([0.0, 2.0, 7.0]), dim=-1)
    expected = (probs * (probs * 3).log()).sum()
    torch.testing.assert_close(kl[0], torch.tensor(0.0))
    torch.testing.assert_close(kl[1], expected)
    kl.sum().backward()
    assert qvalues.grad is not None and torch.isfinite(qvalues.grad).all()
    assert qvalues.grad[0, 2] == 0


def test_configuration_is_validated():
    env = env_config()
    qnetwork = qnetworks.QMLP.from_env(env, hidden_sizes=(8,))
    mixer = mixers.QMix.from_env(env, embed_size=8, hypernet_embed_size=8)
    with pytest.raises(ValueError, match="mixer"):
        algos.MASER(qnetwork, EpisodeMemory(10))
    recurrent = qnetworks.QRNN.from_env(env, mlp_head_sizes=(16,), mlp_tail_sizes=(16,))
    with pytest.raises(ValueError, match="episode replay"):
        algos.MASER(recurrent, TransitionMemory(10), mixer=mixer)
    with pytest.raises(ValueError, match="alpha"):
        algos.MASER(qnetwork, EpisodeMemory(10), mixer=mixer, alpha=1.5)


@pytest.mark.parametrize("recurrent", [False, True])
def test_train_updates_every_network_on_padded_episodes(recurrent: bool):
    torch.manual_seed(0)
    trainer = make_trainer(recurrent=recurrent, lr=1e-2, grad_norm_clipping=10.0)
    batch = padded_batch(trainer)
    assert batch.masked_indices.any()
    online = {
        name: [p.detach().clone() for p in net.parameters()]
        for name, net in (("qnetwork", trainer.qnetwork), ("mixer", trainer.mixer), ("representation", trainer.representation))
    }
    targets = [p.detach().clone() for p in trainer.qtarget.parameters()]

    logs = trainer.train(0, batch)

    expected_keys = {"td-loss", "individual-loss", "correction-loss", "representation-loss", "intrinsic-reward", "subgoal-timestep"}
    assert expected_keys <= logs.keys()
    assert all(math.isfinite(value) for value in logs.values())
    assert logs["intrinsic-reward"] <= 0
    assert logs["correction-loss"] >= 0
    assert 0 <= logs["subgoal-timestep"] < 8
    for name, net in (("qnetwork", trainer.qnetwork), ("mixer", trainer.mixer), ("representation", trainer.representation)):
        changed = any(not torch.equal(before, after) for before, after in zip(online[name], net.parameters(), strict=True))
        assert changed, f"{name} was not updated"
    for before, after in zip(targets, trainer.qtarget.parameters(), strict=True):
        torch.testing.assert_close(before, after)


@pytest.mark.parametrize("mixer_type", [mixers.VDN, mixers.QPlex])
def test_other_mixers_can_be_used(mixer_type):
    trainer = make_trainer(mixer=mixer_type.from_env(env_config()))
    batch = padded_batch(trainer)
    logs = trainer.train(0, batch)
    assert all(math.isfinite(value) for value in logs.values())
    obs, state = env_config().make().reset()
    assert math.isfinite(trainer.value(obs, state))


def test_transition_replay_selects_goals_and_updates_networks():
    torch.manual_seed(0)
    trainer = make_trainer(transition_memory=True, lr=1e-2)
    episode = collect_episode(trainer, time_limit=8)
    batch = TransitionBatch(list(episode.transitions())[:4])
    before = [[p.detach().clone() for p in net.parameters()] for net in (trainer.qnetwork, trainer.mixer, trainer.representation)]
    logs = trainer.train(0, batch)
    assert all(math.isfinite(value) for value in logs.values())
    assert logs["correction-loss"] == 0
    assert "subgoal-timestep" not in logs
    assert 0 <= logs["subgoal-index"] < batch.size
    for old, net in zip(before, (trainer.qnetwork, trainer.mixer, trainer.representation), strict=True):
        assert any(not torch.equal(a, b) for a, b in zip(old, net.parameters(), strict=True))


def test_representation_loss_does_not_backpropagate_into_the_value_networks():
    trainer = make_trainer()
    batch = padded_batch(trainer)
    networks = (trainer.qnetwork, trainer.mixer, trainer.representation)
    initial_states = [deepcopy(net.state_dict()) for net in networks]

    def value_gradients(representation_loss_weight: float):
        for net, state in zip(networks, initial_states, strict=True):
            net.load_state_dict(state)
        trainer.representation_loss_weight = representation_loss_weight
        trainer.train(0, batch)
        return [p.grad.clone() for net in networks[:2] for p in net.parameters() if p.grad is not None]

    without_representation = value_gradients(0.0)
    with_representation = value_gradients(100.0)
    assert len(without_representation) > 0
    for expected, actual in zip(without_representation, with_representation, strict=True):
        torch.testing.assert_close(expected, actual)
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in trainer.representation.parameters())


def test_serialization_round_trip():
    trainer = make_trainer(alpha=0.3, intrinsic_weight=0.1, correction_loss_weight=0.2)
    restored = Trainer.from_dict(trainer.to_dict())
    assert isinstance(restored, MASER)
    assert (restored.alpha, restored.intrinsic_weight, restored.correction_loss_weight) == (0.3, 0.1, 0.2)
    assert restored.name == trainer.name


def test_save_and_load_restore_the_representation(tmp_path):
    trainer = make_trainer()
    trainer.save(tmp_path)
    other = make_trainer()
    other.load(tmp_path)
    for expected, actual in zip(trainer.representation.parameters(), other.representation.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)


@pytest.mark.parametrize("transition_memory", [False, True])
def test_training_loop_smoke(transition_memory: bool):
    env = env_config(time_limit=10).make()
    trainer = make_trainer(transition_memory=transition_memory)
    agent = trainer.make_agent()
    obs, state = env.reset()
    episode = Episode.new(obs, state)
    episode_num = 0
    logs: dict[str, float] = {}
    for time_step in range(1, 81):
        action = agent.choose_action(obs)
        step = env.step(action.action)
        transition = Transition.from_step(obs, state, action, step)
        logs.update(trainer.update_step(transition, time_step))
        episode.add(transition)
        obs, state = step.obs, step.state
        if step.done or step.truncated:
            episode_num += 1
            logs.update(trainer.update_episode(episode, episode_num, time_step))
            obs, state = env.reset()
            episode = Episode.new(obs, state)
            agent.new_episode()
    assert "td-loss" in logs
    assert all(math.isfinite(value) for value in logs.values())
