from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from marlenv import Episode, Observation, State, Transition

from marl.agents import Haven
from marl.algos.dqn import DQN
from marl.algos.haven import HavenTrainer
from marl.models import Action, Agent, EpisodeMemory, TransitionMemory
from marl.models.batch import EpisodeBatch
from marl.nn.mixers.qmix import QMix
from marl.nn.mixers.vdn import VDN
from marl.nn.model_bank.qnetworks import QMLP, QRNN


def make_trainer(recurrent=False, qmix=False, **kwargs):
    network = QRNN if recurrent else QMLP

    def child(extras):
        mixer = QMix(2, 1, 0, embed_size=4, hypernet_embed_size=4) if qmix else VDN()
        return DQN(
            network(2, 2, (1,), (extras,), duelling=False),
            EpisodeMemory(8),
            mixer=mixer,
            batch_size=2,
            gamma=0.5,
            train_interval=(1, "episode"),
        )

    return HavenTrainer(child(3), child(4), 2, 2, 3, 1, 1, **kwargs)


def make_episode(length=5, done=True):
    def obs(t):
        return Observation(np.full((2, 1), t, np.float32), np.ones((2, 2), bool), np.zeros((2, 4), np.float32))

    transitions = [
        Transition(
            obs(t),
            State(np.array([t], np.float32)),
            np.array([0, 1]),
            float(t + 1),
            done and t == length - 1,
            {},
            obs(t + 1),
            State(np.array([t + 1], np.float32)),
            not done and t == length - 1,
            meta_actions=np.full(2, (t // 3) % 2, dtype=np.int64),
        )
        for t in range(length)
    ]
    return Episode.from_transitions(transitions)


@pytest.mark.parametrize("length", [1, 3, 5, 6])
@pytest.mark.parametrize("done", [True, False])
def test_macro_intervals_include_boundaries_and_final_rewards(length, done):
    trainer = make_trainer()
    episode = make_episode(length, done)
    meta = trainer._build_meta_episode(episode)
    assert len(meta) == (length + 2) // 3
    np.testing.assert_array_equal(np.array(meta.all_states).ravel(), list(range(0, length, 3)) + [length])
    expected = [sum(range(t + 1, min(t + 3, length) + 1)) for t in range(0, length, 3)]
    np.testing.assert_array_equal(np.array(meta.rewards).ravel(), expected)
    np.testing.assert_array_equal(meta.actions, episode["meta_actions"][::3])
    assert meta.is_done == done and meta.is_truncated == (not done)
    assert all(x.shape == (2, 3) for x in meta.all_extras)
    np.testing.assert_array_equal(meta.all_extras[0], np.zeros((2, 3)))
    np.testing.assert_array_equal(meta.all_extras[1][:, -2:], [[1, 0], [1, 0]])
    assert all(x.all() for x in meta.all_available_actions)


def test_worker_replay_has_current_and_next_goals_without_mutating_source():
    trainer = make_trainer()
    source = make_episode()
    worker = trainer._worker_episode(source)
    np.testing.assert_array_equal(worker.all_extras[0][:, -2:], [[1, 0], [1, 0]])
    np.testing.assert_array_equal(worker.next_extras[2][:, -2:], [[0, 1], [0, 1]])
    assert all(not extras.any() for extras in source.all_extras)


def test_intrinsic_reward_is_fresh_detached_masked_and_divided_by_k(monkeypatch):
    trainer = make_trainer()
    episodes = [trainer._worker_episode(make_episode()), trainer._worker_episode(make_episode(1, False))]
    worker = EpisodeBatch(episodes)
    meta = EpisodeBatch([trainer._build_meta_episode(e) for e in episodes])
    values = torch.tensor([[2.0, 4.0], [8.0, 10.0], [100.0, 0.0]], requires_grad=True)
    monkeypatch.setattr(trainer, "_values", lambda _: values)
    reward = trainer._intrinsic_rewards(worker, meta)
    # First interval: (6 + .5*8 - 2)/3; terminal interval: (9 - 8)/3.
    torch.testing.assert_close(reward[:, 0], torch.tensor([8 / 3, 8 / 3, 8 / 3, 1 / 3, 1 / 3]))
    # Truncation bootstraps; padding receives no intrinsic reward.
    torch.testing.assert_close(reward[:, 1], torch.tensor([2 / 3, 0.0, 0.0, 0.0, 0.0]))
    assert not reward.requires_grad
    with torch.no_grad():
        values[0] += 3
    assert not torch.equal(reward, trainer._intrinsic_rewards(worker, meta))
    assert worker.rewards[0, 0] == 1


def test_value_target_uses_online_macro_q_and_no_terminal_bootstrap(monkeypatch):
    trainer = make_trainer()
    batch = EpisodeBatch([trainer._build_meta_episode(make_episode())])
    q = torch.tensor([[[[1.0, 3.0], [2.0, 4.0]]]]).expand(3, 1, 2, 2)
    monkeypatch.setattr(trainer.meta_trainer.qnetwork, "batch_qvalues", lambda *a, **kw: q)
    monkeypatch.setattr(trainer.meta_trainer.qtarget, "batch_qvalues", lambda *a, **kw: q * 100)
    torch.testing.assert_close(trainer._value_targets(batch), torch.tensor([[9.5], [9.0]]))


@pytest.mark.parametrize("length", [1, 3])
def test_time_limit_bootstrap_uses_correct_macro_goal(length, monkeypatch):
    trainer = make_trainer()
    episode = trainer._worker_episode(make_episode(length, False))
    batch = EpisodeBatch([episode])
    q = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]]).expand(2, 1, 2, 2)
    monkeypatch.setattr(trainer.meta_trainer.qnetwork, "batch_qvalues", lambda *a, **kw: q)
    trainer._worker_batch(batch)
    goal = torch.tensor([[0.0, 1.0], [0.0, 1.0]]) if length == 3 else torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    torch.testing.assert_close(batch.next_extras[-1, 0, :, -2:], goal)
    torch.testing.assert_close(batch.all_extras[-1, 0, :, -2:], goal)
    np.testing.assert_array_equal(episode.all_extras[-1][:, -2:], [[1.0, 0.0], [1.0, 0.0]])


def test_k_one_has_one_macro_transition_per_primitive_step():
    trainer = replace(make_trainer(), k=1)
    episode = make_episode()
    meta = trainer._build_meta_episode(episode)
    assert len(meta) == len(episode)
    np.testing.assert_array_equal(meta.rewards, episode.rewards)


def test_recurrent_acting_state_survives_replay_and_evaluation():
    trainer = make_trainer(recurrent=True)
    agent = trainer.make_agent()
    obs = next(make_episode().transitions()).obs
    agent.choose_action(obs)
    networks = [trainer.meta_trainer.qnetwork.rnn, trainer.worker_trainer.qnetwork.rnn]
    hidden = [net._hidden_states.clone() for net in networks]
    trainer.update_episode(make_episode(), 0, 5)
    trainer.update_episode(make_episode(1), 1, 6)
    for net, expected in zip(networks, hidden):
        torch.testing.assert_close(net._hidden_states, expected)
    agent.set_testing()
    agent.new_episode()
    agent.choose_action(obs)
    agent.set_training()
    for net, expected in zip(networks, hidden):
        torch.testing.assert_close(net._hidden_states, expected)


@pytest.mark.parametrize("recurrent,qmix", [(False, False), (True, True)])
def test_all_three_estimators_train_on_padded_replay(recurrent, qmix):
    trainer = make_trainer(recurrent, qmix)
    trainer.update_episode(make_episode(), 0, 5)
    groups = [
        list(trainer.meta_trainer.qnetwork.parameters()),
        list(trainer.worker_trainer.qnetwork.parameters()),
        list(trainer.value_network.parameters()),
    ]
    before = [[p.detach().clone() for p in group] for group in groups]
    logs = trainer.update_episode(make_episode(1, False), 1, 6)
    assert {"meta-td-loss", "worker-td-loss", "value-loss", "intrinsic-reward"} <= logs.keys()
    assert all(np.isfinite(v) for v in logs.values())
    for group, old in zip(groups, before):
        assert any(not torch.equal(p, previous) for p, previous in zip(group, old))
    assert len(trainer.worker_trainer.memory) == len(trainer.meta_trainer.memory) == 2
    assert trainer.worker_trainer.memory[0].rewards[0] == 1
    # V updates never backpropagate into either Q policy.
    for group in groups:
        for p in group:
            p.grad = None
    trainer._train_value(trainer.meta_trainer.memory.as_batch())
    assert all(p.grad is None for group in groups[:2] for p in group)


class ScriptedAgent(Agent):
    def __init__(self):
        super().__init__()
        self.calls = []

    def choose_action(self, observation, *, with_details=False):
        self.calls.append((deepcopy(observation), with_details))
        return Action(np.full(2, int(self.is_testing)), marker=np.array([with_details]))


def test_agent_cadence_evaluation_restore_and_observation_ownership():
    meta, worker = ScriptedAgent(), ScriptedAgent()
    agent = Haven(meta, worker, 2, 2, 3, 1, 1)
    obs = next(make_episode().transitions()).obs
    first = agent.choose_action(obs, with_details=True)
    agent.set_testing()
    agent.set_testing()
    agent.new_episode()
    agent.choose_action(obs)
    agent.set_training()
    agent.set_training()
    for _ in range(2):
        np.testing.assert_array_equal(agent.choose_action(obs).meta_actions, [0, 0])
    assert len(meta.calls) == 2
    agent.choose_action(obs)
    assert len(meta.calls) == 3
    assert not obs.extras.any()
    np.testing.assert_array_equal(first.meta_actions, [0, 0])
    assert worker.calls[0][1]
    np.testing.assert_array_equal(worker.calls[0][0].extras[:, -2:], [[1, 0], [1, 0]])
    agent.new_episode()
    agent.choose_action(obs)
    assert len(meta.calls) == 4


def test_warmup_collects_complete_intervals_and_respects_step_schedule():
    trainer = make_trainer(n_meta_warmup_steps=10)
    trainer.update_episode(make_episode(), 0, 5)
    assert trainer.update_episode(make_episode(), 1, 9) == {}
    assert trainer.update_step(next(make_episode().transitions()), 10) == {}
    logs = trainer.update_episode(make_episode(), 2, 14)
    assert "value-loss" in logs
    for child in (trainer.meta_trainer, trainer.worker_trainer):
        child._update_on_steps, child._update_on_episodes, child._train_every_n = True, False, 2
    assert trainer.update_step(next(make_episode().transitions()), 15) == {}
    assert "worker-td-loss" in trainer.update_step(next(make_episode().transitions()), 16)


def test_checkpoint_and_configuration_roundtrip(tmp_path):
    trainer = make_trainer(qmix=True)
    trainer.update_episode(make_episode(), 0, 5)
    trainer.update_episode(make_episode(1), 1, 6)
    restored = HavenTrainer.from_json(trainer.to_json())
    trainer.save(tmp_path / "trainer")
    restored.load(tmp_path / "trainer")
    for actual, expected in zip(restored.networks(), trainer.networks()):
        for key, tensor in expected.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[key], tensor)
    assert restored.value_optimiser.state
    agent = trainer.make_agent()
    agent.save(tmp_path / "agent")
    other = make_trainer(qmix=True).make_agent()
    other.load(tmp_path / "agent")
    for actual, expected in zip(other.networks(), agent.networks()):
        for key, tensor in expected.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[key], tensor)


def test_rejects_invalid_hierarchy_and_unaligned_replay():
    trainer = make_trainer()
    with pytest.raises(ValueError, match="positive"):
        replace(trainer, k=0)
    trainer.meta_trainer.memory = TransitionMemory(8)
    with pytest.raises(TypeError, match="EpisodeMemory"):
        trainer.__post_init__()
