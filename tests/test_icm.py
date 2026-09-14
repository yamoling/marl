from copy import deepcopy

import numpy as np
import pytest
import torch
from marlenv import Episode, Transition
from marlenv.catalog import DiscreteMockEnv

from marl.algos.intrinsic_reward.icm import ICM
from marl.models.batch import EpisodeBatch, TransitionBatch
from marl.nn.model_bank.generic import MLP
from marl.utils import Schedule

N_AGENTS = 2
N_ACTIONS = 3
N_FEATURES = 4


def make_env(n_steps: int = 3):
    return DiscreteMockEnv(n_agents=N_AGENTS, n_actions=N_ACTIONS, end_game=n_steps)


def collect(env, n_steps: int) -> list[Transition]:
    obs, state = env.reset()
    transitions = []
    for _ in range(n_steps):
        action = env.sample_action()
        step = env.step(action)
        transitions.append(Transition.from_step(obs, state, action, step))
        obs, state = step.obs, step.state
    return transitions


def transition_batch(n_steps: int = 3):
    env = make_env(n_steps)
    return env, TransitionBatch(collect(env, n_steps))


def episode_batch(lengths=(2, 3)):
    """Padded batch of episodes: the second one is one step longer than the first."""
    env = make_env(max(lengths))
    transitions = collect(env, max(lengths))
    episodes = []
    for length in lengths:
        ep = Episode.new(transitions[0].obs, transitions[0].state)
        for t in transitions[:length]:
            ep.add(deepcopy(t))
        ep.is_truncated = length != max(lengths)
        episodes.append(ep)
    return env, EpisodeBatch(episodes)


def action_revealing_batch(n_steps: int = 8):
    """
    A batch whose next observation reveals the action that each agent took, so that the inverse model
    has something to learn: `DiscreteMockEnv` observations are a step counter and are action-independent.
    """
    env, batch = transition_batch(n_steps)
    obs = torch.rand(n_steps, N_AGENTS, env.observation_shape[0])
    next_obs = obs.clone()
    next_obs[..., :N_ACTIONS] += torch.nn.functional.one_hot(batch.actions.long(), N_ACTIONS).float()
    batch.obs = obs
    batch.next_obs = next_obs
    return env, batch


def make_icm(env, **kwargs):
    encoder = MLP((N_FEATURES,), env.observation_shape[0], env.extras_shape[0], hidden_sizes=(8,))
    icm = ICM(encoder, env.n_agents, env.n_actions, n_features=N_FEATURES, hidden_size=8, **kwargs)
    icm.randomize()
    return icm


def test_curiosity_is_per_agent_when_the_rewards_are():
    env, batch = transition_batch()
    icm = make_icm(env)
    # Without a mixer, the trainer expands the rewards over the agents and expects an agent-wise signal.
    assert icm.compute(batch.for_individual_learners()).shape == (batch.size, N_AGENTS)


def test_curiosity_is_averaged_over_agents_for_team_rewards():
    env, batch = transition_batch()
    icm = make_icm(env)
    individual = icm.compute(deepcopy(batch).for_individual_learners())
    team = icm.compute(batch)
    assert team.shape == (batch.size,)
    torch.testing.assert_close(team, individual.mean(-1))


def test_curiosity_matches_equation_6():
    env, batch = transition_batch()
    icm = make_icm(env, weight=Schedule.constant(0.5))
    batch = batch.for_individual_learners()
    with torch.no_grad():
        features, next_features = icm.encode(batch)
        one_hot = torch.nn.functional.one_hot(batch.actions.long(), N_ACTIONS).float()
        predicted = icm.forward_model.forward(torch.cat((features, one_hot), dim=-1))
        expected = 0.5 * 0.5 * (predicted - next_features).square().sum(-1)
    torch.testing.assert_close(icm.compute(batch), expected)


def test_padded_time_steps_get_no_curiosity():
    env, batch = episode_batch()
    icm = make_icm(env)
    intrinsic = icm.compute(batch.for_individual_learners())
    assert intrinsic.shape == (3, 2, N_AGENTS)
    # The first episode is one step shorter than the second one.
    assert intrinsic[2, 0].abs().sum().item() == 0.0
    assert intrinsic[2, 1].abs().sum().item() > 0.0


def test_inverse_loss_is_the_masked_cross_entropy_of_the_logits():
    env, batch = episode_batch()
    icm = make_icm(env)
    batch = batch.for_individual_learners()
    with torch.no_grad():
        features, next_features = icm.encode(batch)
        logits = icm.inverse_model.forward(torch.cat((features, next_features), dim=-1))
        errors = torch.nn.functional.cross_entropy(
            logits.reshape(-1, N_ACTIONS), batch.actions.long().reshape(-1), reduction="none"
        ).view_as(batch.actions)
        masks = batch.masks[..., 0].unsqueeze(-1)
        expected = (errors * masks).sum() / (masks.sum() * N_AGENTS)
    assert icm.update(batch, 0)["icm-inverse-loss"] == pytest.approx(expected.item(), rel=1e-5)


def test_total_loss_weighs_the_two_models_with_beta():
    env, batch = transition_batch()
    icm = make_icm(env, beta=0.25)
    logs = icm.update(batch.for_individual_learners(), 0)
    expected = 0.75 * logs["icm-inverse-loss"] + 0.25 * logs["icm-forward-loss"]
    assert logs["ir-loss"] == pytest.approx(expected, rel=1e-5)


def test_update_trains_the_encoder_and_both_models():
    env, batch = transition_batch()
    icm = make_icm(env)
    before = {name: p.detach().clone() for name, p in icm.named_parameters()}
    logs = icm.update(batch.for_individual_learners(), 0)
    assert np.isfinite(logs["ir-loss"])
    changed = {name for name, p in icm.named_parameters() if not torch.equal(p, before[name])}
    assert any(name.startswith("_feature_encoder") for name in changed)
    assert any(name.startswith("inverse_model") for name in changed)
    assert any(name.startswith("forward_model") for name in changed)


def test_inverse_model_learns_a_deterministic_action_mapping():
    """The inverse model should recover the action from two consecutive encodings."""
    env, batch = action_revealing_batch()
    icm = make_icm(env, lr=1e-2)
    batch = batch.for_individual_learners()
    losses = [icm.update(batch, step)["icm-inverse-loss"] for step in range(200)]
    assert losses[-1] < losses[0]
    assert icm.update(batch, 200)["icm-inverse-accuracy"] == pytest.approx(1.0)


def test_curiosity_decreases_on_a_repeated_transition():
    """Curiosity is prediction error: it must fade once the forward model has seen the transition."""
    env, batch = action_revealing_batch(4)
    icm = make_icm(env, lr=1e-2, beta=0.9)
    batch = batch.for_individual_learners()
    before = icm.compute(batch).sum().item()
    for step in range(200):
        icm.update(batch, step)
    assert icm.compute(batch).sum().item() < before


def test_multi_objective_curiosity_is_shared_by_the_objectives():
    env = DiscreteMockEnv(n_agents=N_AGENTS, n_actions=N_ACTIONS, end_game=3, reward_step=[1.0, 2.0])
    batch = TransitionBatch(collect(env, 3))
    batch.multi_objective()
    batch = batch.for_individual_learners()
    icm = make_icm(env)
    intrinsic = icm.compute(batch)
    # One curiosity signal per agent, broadcast over the objectives by the trainer.
    assert intrinsic.shape == (batch.size, N_AGENTS)
    assert np.isfinite(icm.update(batch, 0)["ir-loss"])


def test_encoder_output_size_must_match_the_feature_size():
    env = make_env()
    encoder = MLP((N_FEATURES + 1,), env.observation_shape[0], env.extras_shape[0], hidden_sizes=(8,))
    with pytest.raises(ValueError):
        ICM(encoder, env.n_agents, env.n_actions, n_features=N_FEATURES)


def test_from_env_builds_a_working_module():
    env, batch = transition_batch()
    icm = ICM.from_env(env, n_features=8)
    icm.randomize()
    assert icm.compute(batch.for_individual_learners()).shape == (batch.size, N_AGENTS)


def test_serialization_roundtrip():
    env = make_env()
    icm = make_icm(env, beta=0.3, lr=5e-4, weight=Schedule.linear(1.0, 0.1, 1000))
    restored = ICM.from_json(icm.to_json())
    assert isinstance(restored, ICM)
    assert (restored.n_agents, restored.n_actions, restored.n_features) == (N_AGENTS, N_ACTIONS, N_FEATURES)
    assert (restored.beta, restored.lr) == (0.3, 5e-4)
    assert restored.weight.value == icm.weight.value
    # JSON has no tuples, so shapes come back as lists: compare the sizes instead.
    assert restored.feature_encoder.output_size == icm.feature_encoder.output_size


def test_checkpoint_roundtrip(tmp_path):
    env, batch = transition_batch()
    icm = make_icm(env)
    icm.update(batch.for_individual_learners(), 0)
    icm.save(tmp_path)
    restored = make_icm(env)
    restored.load(tmp_path)
    torch.testing.assert_close(icm.compute(batch), restored.compute(batch))


def test_device_transfer_keeps_the_encoder_in_sync():
    env = make_env()
    icm = make_icm(env)
    icm.to(torch.device("cpu"))
    # The encoder is both a plain attribute (for serialization) and a registered submodule: the two must
    # remain the same object so that `.to()`, `.parameters()` and `state_dict()` stay consistent.
    assert icm.feature_encoder is icm._modules["_feature_encoder"]
    encoder_params = {id(p) for p in icm.feature_encoder.parameters()}
    assert encoder_params.issubset({id(p) for p in icm.parameters()})
