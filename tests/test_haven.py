from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from marlenv import Episode, Observation, State, Transition

from marl.algos.dqn import DQN
from marl.algos.haven import Haven, HavenTrainer
from marl.models import Action, Agent, EpisodeMemory, TransitionMemory
from marl.models.batch import EpisodeBatch
from marl.nn.mixers.qmix import QMix
from marl.nn.mixers.vdn import VDN
from marl.nn.model_bank.qnetworks import QMLP, QRNN


def test_haven_public_imports_are_consistent():
    """The consolidated package remains available through established convenience exports. @ai-generated"""
    from marl.agents import Haven as AgentExport
    from marl.algos import HavenTrainer as TrainerExport
    from marl.algos.haven import HavenSpec as PackageSpec
    from marl.models import HavenSpec

    assert AgentExport is Haven
    assert TrainerExport is HavenTrainer
    assert HavenSpec is PackageSpec


def make_trainer(recurrent=False, qmix=False, transition_memory=False, memory_size=8, meta_memory_size=None, **kwargs):
    network = QRNN if recurrent else QMLP
    if meta_memory_size is None:
        meta_memory_size = memory_size

    def child(extras, size):
        mixer = QMix(2, 1, 0, embed_size=4, hypernet_embed_size=4) if qmix else VDN()
        return DQN(
            network(2, 2, (1,), (extras,), duelling=False),
            TransitionMemory(size) if transition_memory else EpisodeMemory(size),
            mixer=mixer,
            batch_size=2,
            gamma=0.5,
            train_interval=(1, "episode"),
        )

    return HavenTrainer(child(3, meta_memory_size), child(4, memory_size), 2, 2, 3, 1, 1, **kwargs)


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
    _, meta = trainer.replay.assemble_episode(episode)
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
    worker, _ = trainer.replay.assemble_episode(source)
    np.testing.assert_array_equal(worker.all_extras[0][:, -2:], [[1, 0], [1, 0]])
    np.testing.assert_array_equal(worker.next_extras[2][:, -2:], [[0, 1], [0, 1]])
    assert all(not extras.any() for extras in source.all_extras)


def test_intrinsic_reward_is_fresh_detached_masked_and_divided_by_k(monkeypatch):
    trainer = make_trainer()
    episodes = [trainer.replay.assemble_episode(make_episode())[0], trainer.replay.assemble_episode(make_episode(1, False))[0]]
    worker = EpisodeBatch(episodes)
    meta = EpisodeBatch([trainer.replay.assemble_episode(e)[1] for e in (make_episode(), make_episode(1, False))])
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
    batch = EpisodeBatch([trainer.replay.assemble_episode(make_episode())[1]])
    q = torch.tensor([[[[1.0, 3.0], [2.0, 4.0]]]]).expand(3, 1, 2, 2)
    monkeypatch.setattr(trainer.meta_trainer.qnetwork, "batch_qvalues", lambda *a, **kw: q)
    monkeypatch.setattr(trainer.meta_trainer.qtarget, "batch_qvalues", lambda *a, **kw: q * 100)
    torch.testing.assert_close(trainer._value_targets(batch), torch.tensor([[9.5], [9.0]]))


@pytest.mark.parametrize("length", [1, 3])
def test_time_limit_bootstrap_uses_correct_macro_goal(length, monkeypatch):
    trainer = make_trainer()
    episode = make_episode(length, False)
    trainer.update_episode(episode, 0, length)
    sample = trainer.replay.sample_workers(1, trainer.device)
    q = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]]).expand(2, 1, 2, 2)
    monkeypatch.setattr(trainer.meta_trainer.qnetwork, "batch_qvalues", lambda *a, **kw: q)
    batch, _ = trainer._prepare_worker_sample(sample)
    goal = torch.tensor([[0.0, 1.0], [0.0, 1.0]]) if length == 3 else torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    torch.testing.assert_close(batch.next_extras[-1, 0, :, -2:], goal)
    torch.testing.assert_close(batch.all_extras[-1, 0, :, -2:], goal)
    np.testing.assert_array_equal(episode.all_extras[-1][:, -2:], np.zeros((2, 2)))


def test_k_one_has_one_macro_transition_per_primitive_step():
    trainer = replace(make_trainer(), k=1)
    episode = make_episode()
    _, meta = trainer.replay.assemble_episode(episode)
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
    logs = trainer.update_episode(make_episode(), 2, 14)
    assert "value-loss" in logs
    trainer = make_trainer(transition_memory=True, n_meta_warmup_steps=10)
    for child in (trainer.meta_trainer, trainer.worker_trainer):
        child._update_on_steps, child._update_on_episodes, child._train_every_n = True, False, 2
    episode = make_episode(5)
    transitions = list(episode.transitions())
    for i, transition in enumerate(transitions):
        transition.other["meta_actions"] = episode["meta_actions"][i]
    assert trainer.update_step(transitions[0], 15) == {}
    for transition in transitions[1:4]:
        trainer.update_step(transition, 15)
    assert "worker-td-loss" in trainer.update_step(transitions[4], 16)


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
    with pytest.raises(TypeError, match="two TransitionMemory"):
        trainer.__post_init__()


def collect_episode(trainer, episode):
    logs = {}
    for i, transition in enumerate(episode.transitions()):
        transition.other["meta_actions"] = episode["meta_actions"][i]
        logs.update(trainer.update_step(transition, i))
    return logs


def test_episode_replay_streaming_and_episode_only_collection_match():
    streamed = make_trainer()
    direct = make_trainer()
    episode = make_episode(5, False)
    collect_episode(streamed, episode)
    streamed.update_episode(episode, 0, len(episode))
    direct.update_episode(episode, 0, len(episode))
    assert len(streamed.worker_trainer.memory) == len(streamed.meta_trainer.memory) == 1
    for actual, expected in zip(streamed.worker_trainer.memory[0].transitions(), direct.worker_trainer.memory[0].transitions()):
        np.testing.assert_array_equal(actual.obs.extras, expected.obs.extras)
        np.testing.assert_array_equal(actual.next_obs.extras, expected.next_obs.extras)
    for actual, expected in zip(streamed.meta_trainer.memory[0].transitions(), direct.meta_trainer.memory[0].transitions()):
        np.testing.assert_array_equal(actual.reward, expected.reward)
        np.testing.assert_array_equal(actual.obs.extras, expected.obs.extras)


@pytest.mark.parametrize("length,done,k", [(1, True, 3), (5, True, 3), (6, False, 3), (5, False, 3), (5, True, 1)])
def test_transition_collection_matches_episode_replay(length, done, k):
    trainer = replace(make_trainer(transition_memory=True), k=k)
    episode = make_episode(length, done)
    collect_episode(trainer, episode)
    assert len(trainer.worker_trainer.memory) == length
    expected = list(trainer.replay.assemble_episode(episode)[1].transitions())
    assert len(trainer.meta_trainer.memory) == len(expected)
    for actual, wanted in zip(trainer.meta_trainer.memory, expected):
        for field in ("reward", "action"):
            np.testing.assert_array_equal(getattr(actual, field), getattr(wanted, field))
        for field in ("obs", "next_obs"):
            np.testing.assert_array_equal(getattr(actual, field).extras, getattr(wanted, field).extras)
            np.testing.assert_array_equal(getattr(actual, field).data, getattr(wanted, field).data)
        assert actual.done == wanted.done and actual.truncated == wanted.truncated
    worker, _ = trainer.replay.assemble_episode(episode)
    for i, transition in enumerate(trainer.worker_trainer.memory):
        np.testing.assert_array_equal(transition.obs.extras, worker.all_extras[i])
        np.testing.assert_array_equal(transition.next_obs.extras, worker.all_extras[i + 1])
        assert "haven_macro_transition" not in transition.other
        assert "haven_bootstrap_goal" not in transition.other
    trainer.update_episode(episode, 0, length)
    assert len(trainer.worker_trainer.memory) == length  # No insertion twice.
    assert all(not e.any() for e in episode.all_extras)
    # The next episode starts without inheriting the previous macro action.
    collect_episode(trainer, make_episode(1))
    np.testing.assert_array_equal(trainer.meta_trainer.memory[-1].obs.extras[:, -2:], np.zeros((2, 2)))


def test_transition_boundary_waits_for_recorded_next_goal():
    trainer = make_trainer(transition_memory=True)
    episode = make_episode(5)
    steps = list(episode.transitions())
    for i, step in enumerate(steps):
        step.other["meta_actions"] = episode["meta_actions"][i]
    trainer.update_step(steps[0], 0)
    trainer.update_step(steps[1], 1)
    assert len(trainer.worker_trainer.memory) == len(trainer.meta_trainer.memory) == 0
    trainer.update_step(steps[2], 2)
    assert len(trainer.meta_trainer.memory) == 1
    assert len(trainer.worker_trainer.memory) == 2
    trainer.update_step(steps[3], 3)
    assert len(trainer.worker_trainer.memory) == 3
    np.testing.assert_array_equal(trainer.worker_trainer.memory[2].next_obs.extras[:, -2:], [[0, 1], [0, 1]])


def test_transition_advantage_matches_episode_replay_in_shuffled_order_and_after_eviction(monkeypatch):
    trainer = make_trainer(transition_memory=True, meta_memory_size=1)
    episode = make_episode(5)
    collect_episode(trainer, episode)
    order = [4, 0, 2, 3, 1]
    worker_episode, meta_episode = trainer.replay.assemble_episode(episode)
    episode_batch = EpisodeBatch([worker_episode])
    meta_batch = EpisodeBatch([meta_episode])
    expected = trainer._intrinsic_rewards(episode_batch, meta_batch)[:, 0][order]
    monkeypatch.setattr(np.random, "choice", lambda *args, **kwargs: np.array(order))
    sample = trainer.replay.sample_workers(5, trainer.device)
    raw = sample.workers.rewards.clone()
    actual, _ = trainer._prepare_worker_sample(sample)
    torch.testing.assert_close(actual.rewards, raw + expected)
    assert not actual.rewards.requires_grad
    with torch.no_grad():
        for parameter in trainer.value_network.parameters():
            parameter.add_(0.1)
    refreshed, _ = trainer._prepare_worker_sample(trainer.replay.sample_workers(5, trainer.device))
    assert not torch.equal(refreshed.rewards, actual.rewards)
    np.testing.assert_array_equal([t.reward.item() for t in trainer.worker_trainer.memory], [1, 2, 3, 4, 5])


@pytest.mark.parametrize("length", [1, 3])
def test_transition_time_limit_bootstrap(length, monkeypatch):
    trainer = make_trainer(transition_memory=True)
    collect_episode(trainer, make_episode(length, False))
    monkeypatch.setattr(np.random, "choice", lambda *args, **kwargs: np.array([length - 1]))
    sample = trainer.replay.sample_workers(1, trainer.device)
    monkeypatch.setattr(trainer.meta_trainer.qnetwork, "batch_qvalues", lambda *a, **kw: torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]))
    batch, _ = trainer._prepare_worker_sample(sample)
    expected = [[0.0, 1.0], [0.0, 1.0]] if length == 3 else [[1.0, 0.0], [1.0, 0.0]]
    torch.testing.assert_close(batch.next_extras[0, :, -2:], torch.tensor(expected))


@pytest.mark.parametrize("qmix", [False, True])
def test_transition_replay_trains_all_estimators_on_steps(qmix):
    trainer = make_trainer(transition_memory=True, qmix=qmix)
    for child in (trainer.meta_trainer, trainer.worker_trainer):
        child._update_on_steps, child._update_on_episodes, child._train_every_n = True, False, 1
    groups = [trainer.meta_trainer.qnetwork, trainer.worker_trainer.qnetwork, trainer.value_network]
    before = [[p.detach().clone() for p in net.parameters()] for net in groups]
    logs = collect_episode(trainer, make_episode(5))
    assert {"meta-td-loss", "worker-td-loss", "value-loss", "intrinsic-reward"} <= logs.keys()
    assert all(np.isfinite(v) for v in logs.values())
    for net, old in zip(groups, before):
        assert any(not torch.equal(p, previous) for p, previous in zip(net.parameters(), old))
    restored = HavenTrainer.from_json(trainer.to_json())
    assert isinstance(restored.meta_trainer.memory, TransitionMemory)


def test_transition_replay_rejects_recurrent_q_and_value_networks():
    with pytest.raises(ValueError, match="feed-forward"):
        make_trainer(recurrent=True, transition_memory=True)
    value = QRNN(1, 2, (1,), (3,), duelling=False)
    with pytest.raises(ValueError, match="feed-forward"):
        make_trainer(transition_memory=True, value_network=value)
