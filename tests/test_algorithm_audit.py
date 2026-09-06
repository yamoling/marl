"""Numerical and integration regressions found during the repository audit."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from marlenv import Transition
from marlenv.catalog import DiscreteMockEnv

from marl.algos.dqn import DQN
from marl.algos.ppo import PPO
from marl.algos.qlearning import QLearning
from marl.algos.qplex import QPlex
from marl.models.batch import TransitionBatch
from marl.nn.mixers.qatten import Qatten
from marl.nn.mixers.qplex import QPlex as QPlexMixer
from marl.nn.model_bank import qnetworks
from marl.policy import ArgMax, CategoricalPolicy, EpsilonGreedy


def make_batch(n=3):
    env = DiscreteMockEnv(n_agents=2, n_actions=3, end_game=n)
    obs, state = env.reset()
    transitions = []
    for _ in range(n):
        action = env.sample_action()
        step = env.step(action)
        transitions.append(Transition.from_step(obs, state, action, step))
        obs, state = step.obs, step.state
    return env, TransitionBatch(transitions)


@pytest.mark.parametrize("policy", [ArgMax(), EpsilonGreedy.constant(0), CategoricalPolicy()])
def test_policy_mask_does_not_change_qtable(policy):
    q = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    original = q.copy()
    policy.get_action(q, np.array([[True, False, True]]))
    np.testing.assert_array_equal(q, original)


def test_tabular_update_masks_actions_and_terminal_bootstrap():
    trainer = QLearning(3, 2, lr=1.0, gamma=0.5)
    transition = SimpleNamespace(obs="s", next_obs="next", action=np.array([1, 2]), reward=np.array([2.0]), done=True)
    trainer.update_step(transition, 0)
    np.testing.assert_array_equal(trainer._qtable["s"], [[1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
    _, batch = make_batch()
    transition = batch.transitions[0]
    transition.next_obs.available_actions[:] = [[True, False, True], [True, False, True]]
    trainer._qtable[transition.next_obs][:] = [[3.0, 99.0, 5.0], [7.0, 99.0, 1.0]]
    trainer.update_step(transition, 0)
    actual = trainer._qtable[transition.obs][np.arange(2), transition.action]
    np.testing.assert_allclose(actual, transition.reward.item() + 0.5 * np.array([5.0, 7.0]))


def test_tabular_checkpoint_roundtrip(tmp_path):
    trainer = QLearning(3, 2)
    trainer._qtable["s"][0, 1] = 12.0
    trainer.save(tmp_path)
    restored = QLearning(3, 2)
    restored.load(tmp_path)
    np.testing.assert_array_equal(restored._qtable["s"], trainer._qtable["s"])


def test_mc_bootstrap_keeps_episode_and_agent_dimensions():
    _, batch = make_batch(1)
    batch.rewards = torch.zeros(1, 2, 2)
    batch.dones = torch.zeros(1, 2, 2, dtype=torch.bool)
    batch.masks = torch.ones(1, 2, 2)
    bootstrap = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    torch.testing.assert_close(batch.compute_mc_returns(0.5, bootstrap), 0.5 * bootstrap.unsqueeze(0))


def test_individual_multiobjective_rewards_preserve_objective_order():
    _, batch = make_batch()
    batch.rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    batch.__dict__.pop("reward_size", None)
    batch.dones = torch.zeros(3, 2, dtype=torch.bool)
    batch.masks = torch.ones(3, 2)
    expected = batch.rewards.unsqueeze(-2).expand(3, 2, 2).clone()
    batch.for_individual_learners()
    torch.testing.assert_close(batch.rewards, expected)


def test_ppo_bootstrap_uses_next_extras():
    _, batch = make_batch()
    batch = batch.for_individual_learners()
    batch.extras = torch.zeros(3, 2, 1)
    batch.next_extras = torch.ones(3, 2, 1) * 4
    trainer = SimpleNamespace(
        critic=SimpleNamespace(value=lambda obs, extras: extras[..., 0].clone()),
        mixer=None,
        gamma=0.5,
        gae_lambda=0.0,
        normalize_advantages=False,
    )
    _, advantages = PPO._compute_training_data(trainer, batch)
    torch.testing.assert_close(advantages, batch.rewards + 2 * batch.not_dones)


@pytest.mark.parametrize("double", [False, True])
def test_dqn_terminal_without_available_actions_is_finite(double):
    env, batch = make_batch()
    trainer = DQN(qnetworks.from_env(env, hidden_sizes=(8,)), double_qlearning=double)
    batch = batch.for_individual_learners()
    batch.next_available_actions[-1] = False
    with torch.no_grad():
        targets = trainer._compute_qtargets(batch)
    assert torch.isfinite(targets).all()
    torch.testing.assert_close(targets[-1], batch.rewards[-1])


@pytest.mark.parametrize("double", [False, True])
def test_qplex_trainer_can_compute_and_differentiate_loss(double):
    env, batch = make_batch()
    trainer = QPlex(qnetworks.from_env(env, hidden_sizes=(8,)), mixer=QPlexMixer.from_env(env), double_qlearning=double)
    logs = trainer.train(0, batch)
    assert np.isfinite(logs["td-loss"])


def test_qatten_episode_batch_equals_flat_batch():
    mixer = Qatten(n_agents=2, state_size=4, state_extras_size=1, unit_dim=2)
    q, states, extras = torch.randn(3, 2, 2), torch.randn(3, 2, 4), torch.randn(3, 2, 1)
    expected = mixer(q.flatten(0, 1), states.flatten(0, 1), extras.flatten(0, 1)).reshape(3, 2)
    torch.testing.assert_close(mixer(q, states, extras), expected)


def test_single_agent_one_hot_keeps_agent_axis():
    _, batch = make_batch()
    batch.actions = torch.tensor([[0], [1], [2]])
    assert batch.one_hot_actions.shape == (3, 1, 3)


def test_dqn_to_preserves_optimizer_moments_and_extra_groups():
    env, batch = make_batch()
    trainer = DQN(qnetworks.from_env(env, hidden_sizes=(8,)))
    trainer.train(0, batch.for_individual_learners())
    extra = torch.nn.Parameter(torch.ones(1))
    trainer.optimiser.add_param_group({"params": [extra]})
    param = next(trainer.qnetwork.parameters())
    expected = trainer.optimiser.state[param]["exp_avg"].clone()
    trainer.to(torch.device("cpu"))
    assert len(trainer.optimiser.param_groups) == 2
    torch.testing.assert_close(trainer.optimiser.state[param]["exp_avg"], expected)


def test_randomized_dqn_starts_with_synchronized_targets():
    env, _ = make_batch()
    trainer = DQN(qnetworks.from_env(env, hidden_sizes=(8,)))
    trainer.randomize()
    for online, target in zip(trainer.qnetwork.parameters(), trainer.qtarget.parameters(), strict=True):
        torch.testing.assert_close(online, target)


def test_gae_does_not_cross_time_limit_reset():
    _, batch = make_batch()
    batch.transitions[0].truncated = True
    batch.rewards = torch.tensor([1.0, 100.0, 100.0])
    values = torch.zeros(3)
    next_values = torch.tensor([4.0, 0.0, 0.0])
    advantages = batch.compute_gae(0.5, values, next_values, trace_decay=1.0)
    assert advantages[0].item() == 3.0


def test_icm_uses_logits_and_supports_episode_dimensions():
    from marl.algos.intrinsic_reward.icm import ICM
    from marl.nn.model_bank.generic import MLP

    _, batch = make_batch()
    module = ICM(MLP((4,), 2, 0, hidden_sizes=(4,)), 2, 3, n_features=4)
    batch.states = torch.zeros(3, 2)
    batch.next_states = torch.ones(3, 2)
    batch.states_extras = batch.next_states_extras = torch.empty(3, 0)
    with torch.no_grad():
        for parameter in module.inverse_model.parameters():
            parameter.zero_()
        module.inverse_model[-1].bias.copy_(torch.tensor([3.0, 0.0, -2.0, 3.0, 0.0, -2.0]))
    expected = torch.nn.functional.cross_entropy(
        torch.tensor([[3.0, 0.0, -2.0]]).expand(6, -1), batch.actions.flatten()
    )
    module.to(torch.device("cpu"))
    logs = module.update(batch, 0)
    assert logs["icm-inverse-loss"] == pytest.approx(expected.item())
    batch.states = batch.states.unsqueeze(1)
    batch.next_states = batch.next_states.unsqueeze(1)
    batch.states_extras = batch.states_extras.unsqueeze(1)
    batch.next_states_extras = batch.next_states_extras.unsqueeze(1)
    batch.actions = batch.actions.unsqueeze(1)
    batch.__dict__.pop("one_hot_actions", None)
    batch.masks = torch.ones(3, 1)
    assert module.compute(batch).shape == (3, 1)
    assert np.isfinite(module.update(batch, 1)["ir-loss"])


def test_rnd_empty_predictor_sample_is_finite_and_does_not_update():
    from marl.algos.intrinsic_reward.random_network_distillation import RND

    _, batch = make_batch()
    module = RND((2,), 0, output_shape=(4,), update_ratio=0.0, n_warmup_steps=0)
    batch.next_states = torch.ones(3, 2)
    batch.next_states_extras = torch.empty(3, 0)
    before = [p.detach().clone() for p in module.parameters()]
    assert module.update(batch, 0)["ir-loss"] == 0.0
    for actual, expected in zip(module.parameters(), before, strict=True):
        torch.testing.assert_close(actual, expected)


def test_ppoc_initialization_honors_step_interval():
    from marl.algos.ppoc import PPOC
    from marl.models import TransitionMemory

    oc = toy_options()
    trainer = PPOC(oc, 2, train_interval=(4, "step"), minibatch_size=2)
    assert isinstance(trainer.memory, TransitionMemory)


def toy_options():
    from marl.models.nn.options import OptionCriticNetwork

    class ToyOptions(OptionCriticNetwork):
        __hash__ = object.__hash__

    oc = ToyOptions((1,), 2)
    oc.register_parameter("weight", torch.nn.Parameter(torch.ones(1)))
    oc.compute_q_options = lambda obs, extras: oc.weight * torch.ones(*obs.shape[:-1], 2)
    oc.value_on_arrival = lambda obs, extras, options: torch.zeros_like(options.squeeze(-1), dtype=torch.float32)
    oc.termination_probability = lambda obs, extras, options: oc.weight.sigmoid().expand_as(options.squeeze(-1))
    return oc


def test_option_critic_optimizer_includes_mixer():
    from marl.algos.option_critic import OptionCritic
    from marl.nn.mixers.qmix import QMix

    mixer = QMix(n_agents=2, state_size=4, state_extras_size=0)
    trainer = OptionCritic(toy_options(), 2, mixer=mixer)
    optimized = {id(p) for group in trainer.optim.param_groups for p in group["params"]}
    assert all(id(p) in optimized for p in mixer.parameters())


def test_ppoc_critic_returns_do_not_depend_on_advantage_normalization():
    from marl.algos.ppoc import PPOC

    _, batch = make_batch()
    batch = batch.for_individual_learners()
    batch.rewards = torch.tensor([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]])
    batch._cache["options"] = torch.zeros_like(batch.actions)
    trainer = SimpleNamespace(
        target_oc=toy_options(), target_mixer=None, gamma=0.9, gae_lambda=0.95, normalize_advantages=False
    )
    expected, _ = PPOC._compute_training_data(trainer, batch)
    trainer.normalize_advantages = True
    actual, _ = PPOC._compute_training_data(trainer, batch)
    torch.testing.assert_close(actual, expected)


def test_nstep_finalizes_successor_and_terminal_tail():
    from marl.models.replay_memory.nstep_memory import NStepMemory

    _, batch = make_batch(4)
    memory = NStepMemory(10, 3, 0.5)
    for t in batch.transitions[:3]:
        memory.add_transition(t)
    assert len(memory) == 1
    first = memory[0]
    np.testing.assert_array_equal(first.next_state.data, batch.transitions[2].next_state.data)
    np.testing.assert_array_equal(first.next_obs.data, batch.transitions[2].next_obs.data)
    assert first.reward.item() == 1.75
    memory.add_transition(batch.transitions[3])
    assert len(memory) == 4
    assert all(memory[i].done for i in range(1, 4))
    assert memory[-1].reward.item() == 1.0
    memory.clear()
    assert len(memory) == 0


def test_prioritized_ring_indices_keep_their_transition_after_eviction(monkeypatch):
    from marl.models import PrioritizedMemory, TransitionMemory

    _, batch = make_batch(4)
    memory = PrioritizedMemory(TransitionMemory(3), multi_objective=False)
    for t in batch.transitions:
        memory.add_transition(t)
    assert len(memory) == 3
    # The tree overwrites physical slot 0; deque slot 0 is instead the oldest item.
    memory.tree = SimpleNamespace(sample=lambda n: ([0], [1.0]), total=3.0)
    sampled = memory.sample(1)
    assert sampled.transitions[0] is batch.transitions[3]


def test_prioritized_individual_loss_and_clear():
    from marl.models import PrioritizedMemory, TransitionMemory

    env, batch = make_batch(4)
    trainer = DQN(qnetworks.from_env(env, hidden_sizes=(8,)))
    trainer.memory = PrioritizedMemory(TransitionMemory(10), multi_objective=False)
    for t in batch.transitions:
        trainer.memory.add(t)
    sampled = trainer.memory.sample(2).for_individual_learners()
    assert np.isfinite(trainer.train(0, sampled)["td-loss"])
    trainer.memory.clear()
    assert len(trainer.memory) == 0
    assert trainer.memory.tree.total == 0


def test_ppo_epoch_visits_each_transition(monkeypatch):
    from marl.nn.model_bank.actor_critics import CategoricalLinearActor, LinearCritic

    env, batch = make_batch(5)
    trainer = PPO(
        CategoricalLinearActor.from_env(env, mlp_sizes=(8,)),
        LinearCritic.from_env(env, mlp_sizes=(8,)),
        None,
        train_interval=(5, "step"),
        minibatch_size=2,
        n_epochs=2,
    )
    for t in batch.transitions:
        trainer.memory.add(t)
    visited = []
    original = TransitionBatch.get_minibatch

    def record(self, indices):
        visited.extend(indices)
        return original(self, indices)

    monkeypatch.setattr(TransitionBatch, "get_minibatch", record)
    trainer.train(0)
    assert len(visited) == 10
    assert sorted(visited[:5]) == sorted(visited[5:]) == list(range(5))


def test_recurrent_training_batches_are_independent_and_preserve_acting_history():
    from marl.nn.model_bank.generic import RNN

    net = RNN((2,), 3, 0, mlp_head_sizes=(4,), mlp_tail_sizes=(4,))
    net(torch.ones(1, 2, 3), torch.empty(1, 2, 0))
    acting_hidden = net._hidden_states.clone()
    obs, extras = torch.randn(3, 2, 2, 3), torch.empty(3, 2, 2, 0)
    first = net(obs, extras)
    second = net(obs, extras)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(net._hidden_states, acting_hidden)
    first.sum().backward()
    net.zero_grad()
    second.sum().backward()


def test_repeated_eval_preserves_recurrent_training_history():
    from marl.nn.model_bank.generic import RNN

    net = RNN((2,), 3, 0, mlp_head_sizes=(4,), mlp_tail_sizes=(4,))
    net(torch.ones(1, 2, 3), torch.empty(1, 2, 0))
    expected = net._hidden_states.clone()
    net.eval()
    net.eval()
    net.train()
    torch.testing.assert_close(net._hidden_states, expected)


def test_qnetwork_softmax_adapter_can_act():
    env, _ = make_batch()
    q = qnetworks.from_env(env, hidden_sizes=(8,))
    actor = q.to_softmax_actor()
    obs, _ = env.reset()
    data, extras, available = obs.as_tensors(torch.device("cpu"), batch_dim=True, actions=True)
    dist = actor.policy(data, extras, available_actions=available)
    torch.testing.assert_close(dist.probs, q.batch_qvalues(data, extras).softmax(-1))


def test_mlp_respects_requested_output_activation():
    from marl.nn.model_bank.generic import MLP

    net = MLP((2,), 3, 0, hidden_sizes=(4,), hidden_activation="relu", output_activation="sigmoid")
    assert isinstance(net.nn[-1], torch.nn.Sigmoid)


def episode_batch():
    from marlenv import Episode

    from marl.models.batch import EpisodeBatch

    _, batch = make_batch(3)
    episodes = []
    for length in (2, 3):
        ep = Episode.new(batch.transitions[0].obs, batch.transitions[0].state)
        from copy import deepcopy

        for t in batch.transitions[:length]:
            ep.add(deepcopy(t))
        ep.is_truncated = length == 2
        episodes.append(ep)
    return EpisodeBatch(episodes)


def test_mc_truncated_episode_uses_its_own_final_value():
    batch = episode_batch().for_individual_learners()
    next_values = torch.full_like(batch.rewards, 4.0)
    result = batch.compute_mc_returns(0.5, next_values[-1], next_values=next_values)
    torch.testing.assert_close(result[:, 0], torch.tensor([[2.5, 2.5], [3.0, 3.0], [0.0, 0.0]]))
    torch.testing.assert_close(result[:, 1], torch.tensor([[1.75, 1.75], [1.5, 1.5], [1.0, 1.0]]))


def test_rnd_episode_statistics_keep_feature_shape_and_ignore_extrinsic_rewards():
    from marl.algos.intrinsic_reward.random_network_distillation import RND

    batch = episode_batch()
    batch.next_states = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    batch.next_states_extras = torch.empty(3, 2, 0)
    module = RND((2,), 0, output_shape=(4,), normalise_rewards=True)
    module._warmup_done = True
    module.forward = lambda states, extras: torch.ones(*states.shape[:-1], 4)
    batch.rewards = torch.zeros_like(batch.rewards)
    reward = module.compute(batch)
    assert torch.isfinite(reward).all()
    assert module._running_states.mean.shape == (2,)
    assert module._running_returns.mean.numel() == 1
    assert module._running_returns.mean.item() > 0
    assert reward[-1, 0] == 0


@pytest.mark.parametrize("duelling", [True, False])
def test_noisy_qmlp_respects_action_count_with_single_hidden_layer(duelling):
    env, batch = make_batch()
    network = qnetworks.from_env(env, hidden_sizes=(8,), noisy=True, duelling=duelling)
    assert network.batch_qvalues(batch.obs, batch.extras).shape == (3, 2, 3)


def test_dqn_checkpoint_keeps_online_and_target_weights_distinct(tmp_path):
    env, _ = make_batch()
    trainer = DQN(qnetworks.from_env(env, hidden_sizes=(8,)), mixer=QPlexMixer.from_env(env))
    with torch.no_grad():
        for p in trainer.qnetwork.parameters():
            p.fill_(1.0)
        for p in trainer.qtarget.parameters():
            p.fill_(2.0)
    trainer.save(tmp_path)
    trainer.randomize()
    trainer.load(tmp_path)
    assert all(torch.all(p == 1.0) for p in trainer.qnetwork.parameters())
    assert all(torch.all(p == 2.0) for p in trainer.qtarget.parameters())


def test_recurrent_acer_can_update_twice():
    from marl.algos.acer import ACER
    from marl.nn.model_bank.actor_critics import CategoricalRecurrentActor

    env, _ = make_batch()
    actor = CategoricalRecurrentActor.from_env(env, independent=False, mlp_head_sizes=(8,), mlp_tail_sizes=(8,))
    critic = qnetworks.from_env(env, recurrent=True, mlp_head_sizes=(8,), mlp_tail_sizes=(8,))
    trainer = ACER(actor, critic, None)
    batch = episode_batch()
    for ep in batch.episodes:
        ep.other["action_probabilities"] = [np.full((2, 3), 1 / 3, dtype=np.float32) for _ in ep.actions]
    for _ in range(2):
        assert np.isfinite(trainer._update(batch, 0, on_policy=True)["loss"])


def test_ppoc_termination_handles_individual_episode_axes():
    from marl.algos.ppoc import PPOC

    batch = episode_batch().for_individual_learners()
    trainer = SimpleNamespace(oc=toy_options(), target_oc=toy_options(), target_mixer=None, termination_reg=0.01)
    options = torch.zeros_like(batch.actions).unsqueeze(-1)
    loss = PPOC._compute_termination_loss(trainer, batch, options)
    expected = torch.sigmoid(torch.tensor(1.0)) * 0.01 * batch.not_dones.mul(batch.masks).sum() / batch.n_items
    torch.testing.assert_close(loss, expected)


def test_training_episode_respects_exact_run_budget():
    from marl.runners.simple_runner import _train_episode

    env, _ = make_batch(20)
    trainer = DQN(qnetworks.from_env(env, hidden_sizes=(8,)))
    run = SimpleNamespace(
        n_steps=3,
        should_test_at=lambda t: False,
        logger=SimpleNamespace(log_training_data=lambda *args: None, log_train=lambda *args: None),
    )
    episode = _train_episode(env, env, trainer.make_agent(), trainer, 0, 0, False, True, run)
    assert len(episode) == 3
    assert episode.is_truncated


@pytest.mark.parametrize("kind", ["advantage", "potential"])
def test_intrinsic_potential_has_no_terminal_bootstrap(kind):
    from marl.algos.intrinsic_reward.advantage_ir import AdvantageIntrinsicReward
    from marl.algos.intrinsic_reward.value_ir import ValuePotentialIntrinsicReward

    _, batch = make_batch()
    module = SimpleNamespace(
        network=SimpleNamespace(value=lambda *args: torch.ones(3)),
        target_network=SimpleNamespace(value=lambda *args: torch.full((3,), 10.0)),
        gamma=0.5,
    )
    if kind == "advantage":
        actual = AdvantageIntrinsicReward.compute(module, batch)
        expected = batch.rewards - 1 + 5 * batch.not_dones
    else:
        actual = ValuePotentialIntrinsicReward.compute(module, batch)
        expected = -1 + 5 * batch.not_dones
    torch.testing.assert_close(actual, expected.float())


def test_qplex_target_actions_follow_online_argmax_in_double_q(monkeypatch):
    env, batch = make_batch()
    trainer = QPlex(qnetworks.from_env(env, hidden_sizes=(8,)), mixer=QPlexMixer.from_env(env))
    shape = (4, 2, 3)
    online = torch.tensor([1.0, 3.0, 2.0]).expand(shape).clone()
    target = torch.tensor([9.0, 4.0, 1.0]).expand(shape).clone()
    monkeypatch.setattr(trainer.qnetwork, "batch_qvalues", lambda *args, **kwargs: online)
    monkeypatch.setattr(trainer.qtarget, "batch_qvalues", lambda *args, **kwargs: target)
    captured = {}

    def mix(q, *args, **kwargs):
        captured.update(kwargs)
        torch.testing.assert_close(q, torch.full((3, 2), 4.0))
        return q.sum(-1)

    monkeypatch.setattr(trainer.target_mixer, "forward", mix)
    trainer._compute_qtargets(batch)
    assert torch.all(captured["one_hot_actions"].argmax(-1) == 1)


def test_dqn_uses_nstep_discount():
    from marl.models.replay_memory.nstep_memory import NStepMemory

    env, batch = make_batch(4)
    memory = NStepMemory(10, 3, 0.5)
    for i, t in enumerate(batch.transitions[:3]):
        if i == 2:
            t.done = True
        memory.add(t)
    trainer = DQN(qnetworks.from_env(env, hidden_sizes=(8,)), gamma=0.5, double_qlearning=False)
    trainer.qtarget.batch_qvalues = lambda *args, **kwargs: torch.full((2, 2, 3), 8.0)
    batch = memory.get_batch([0, 1, 2])

    targets = trainer._compute_qtargets(memory.sample(1).for_individual_learners())
    torch.testing.assert_close(targets, torch.full((1, 2), 2.75))


def test_multiobjective_dqn_preserves_rewards_and_selects_one_joint_action():
    env, batch = make_batch()
    for transition in batch.transitions:
        transition.reward = np.array([1.0, 2.0], dtype=np.float32)
    network = qnetworks.from_env(env, hidden_sizes=(8,), n_objectives=2, duelling=False)
    trainer = DQN(network, double_qlearning=False, gamma=0.5)
    batch = batch.for_individual_learners()
    utilities = torch.tensor([[9.0, 0.0], [0.0, 10.0], [1.0, 1.0]]).expand(4, 2, 3, 2).clone()
    trainer.qtarget.batch_qvalues = lambda *args, **kwargs: utilities
    targets = trainer._compute_qtargets(batch)
    torch.testing.assert_close(targets[0], torch.tensor([[1.0, 7.0], [1.0, 7.0]]))
    assert np.isfinite(trainer.train(0, batch)["td-loss"])


def test_rnd_accepts_image_states():
    from marl.algos.intrinsic_reward.random_network_distillation import RND

    module = RND((3, 9, 9), 0, output_shape=(4,))
    assert module.forward(torch.zeros(2, 3, 9, 9), torch.empty(2, 0)).shape == (2, 4)


def test_ppoc_value_loss_trains_online_mixer_only():
    from copy import deepcopy

    from marl.algos.ppoc import PPOC
    from marl.nn.mixers.qmix import QMix

    env, batch = make_batch()
    mixer = QMix.from_env(env)
    trainer = SimpleNamespace(oc=toy_options(), mixer=mixer, target_mixer=deepcopy(mixer))
    options = torch.zeros_like(batch.actions).unsqueeze(-1)
    PPOC._compute_critic_loss(trainer, batch, torch.zeros_like(batch.rewards), options).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mixer.parameters())
    assert all(p.grad is None for p in trainer.target_mixer.parameters())


def test_ppo_and_acer_to_preserve_optimizer_history():
    from marl.algos.acer import ACER
    from marl.nn.model_bank.actor_critics import CategoricalLinearActor, LinearCritic

    env, _ = make_batch()
    actor = CategoricalLinearActor.from_env(env, mlp_sizes=(8,))
    critic = LinearCritic.from_env(env, mlp_sizes=(8,))
    for trainer in (PPO(actor, critic, None), ACER(actor, qnetworks.from_env(env, hidden_sizes=(8,)), None)):
        loss = sum(p.square().sum() for group in trainer._optimizer.param_groups for p in group["params"])
        trainer._optimizer.zero_grad()
        loss.backward()
        trainer._optimizer.step()
        parameter = next(actor.parameters())
        expected = trainer._optimizer.state[parameter]["exp_avg"].clone()
        trainer.to(torch.device("cpu"))
        torch.testing.assert_close(trainer._optimizer.state[parameter]["exp_avg"], expected)
