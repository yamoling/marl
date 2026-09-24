import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from marlenv import Episode, Transition

from marl import algos, policy
from marl.algos.maven.mutual_information_trainer import MITrainer
from marl.env import LLEConfig
from marl.models import Trainer
from marl.models.agent.agent_wrapper import AgentWrapper
from marl.nn.model_bank import MAVENQnetwork
from marl.runners.simple_runner import _train_episode, seeded_rollout

NOISE_SIZE = 3


def test_generic_extra_padding_and_legacy_configs():
    plain = LLEConfig(6)
    padded = LLEConfig(6, extra_padding_size=NOISE_SIZE)
    assert padded.extras_shape == (plain.extras_shape[0] + NOISE_SIZE,)
    assert padded.noise_size == NOISE_SIZE
    assert padded.extras_meanings[-NOISE_SIZE:] == ["padding-0", "padding-1", "padding-2"]
    restored = LLEConfig.from_json(padded.to_json())
    assert restored.extra_padding_size == NOISE_SIZE
    assert restored.extras_shape == padded.extras_shape

    legacy_data = env_config().to_dict()
    legacy_data.pop("extra_padding_size")
    legacy = LLEConfig.from_dict(legacy_data)
    assert legacy.maven_noise_size == NOISE_SIZE
    assert legacy.noise_size == NOISE_SIZE
    assert legacy.extras_shape == padded.extras_shape
    with pytest.raises(ValueError, match="Specify only one"):
        LLEConfig(6, extra_padding_size=NOISE_SIZE, maven_noise_size=NOISE_SIZE)


def env_config(noise_size=NOISE_SIZE):
    return LLEConfig(6, obs_type="flattened", state_type="flattened", time_limit=8, maven_noise_size=noise_size)


def make_maven(noise_size=NOISE_SIZE, **kwargs) -> algos.MAVEN:
    env = env_config(noise_size)
    qnetwork = MAVENQnetwork.from_env(env, agent_output_size=16)
    return algos.MAVEN(
        qnetwork,
        policy.EpsilonGreedy.constant(0.5),
        env,
        batch_size=2,
        bandit_batch_size=2,
        n_epochs=1,
        qmix_embed_size=8,
        qmix_hypernet_embed_size=8,
        **kwargs,
    )


def test_qnetwork_from_env_matches_environment():
    env = env_config()
    qnetwork = MAVENQnetwork.from_env(env, agent_output_size=16)
    assert qnetwork.n_actions == env.n_actions
    assert qnetwork.n_agents == env.n_agents
    assert qnetwork.obs_shape == env.observation_shape
    assert qnetwork.extras_shape == env.extras_shape
    assert qnetwork.noise_size == NOISE_SIZE


@pytest.mark.parametrize("obs_type", ["flattened", "layered"])
def test_qnetwork_forward_on_transitions_and_episodes(obs_type):
    env = LLEConfig(6, obs_type=obs_type, maven_noise_size=NOISE_SIZE)
    qnetwork = MAVENQnetwork.from_env(env, agent_output_size=16)
    n_agents, n_actions = env.n_agents, env.n_actions
    for dims in [(2,), (3, 2)]:
        obs = torch.rand(*dims, n_agents, *env.observation_shape)
        extras = torch.rand(*dims, n_agents, *env.extras_shape)
        masks = torch.ones(dims)
        qvalues = qnetwork.batch_qvalues(obs, extras, masks=masks)
        assert qvalues.shape == (*dims, n_agents, n_actions)


@pytest.mark.parametrize("z_policy_type", ["return", "uniform"])
def test_training_loop_smoke(z_policy_type):
    env = env_config().make()
    trainer = make_maven(z_policy_type=z_policy_type)
    agent = trainer.make_agent()
    logs: dict[str, float] = {}
    time_step = 0
    for episode_num in range(1, 5):
        obs, state = env.reset()
        agent.new_episode()
        episode = Episode.new(obs, state)
        done = False
        while not done:
            time_step += 1
            action = agent.choose_action(obs)
            step = env.step(action.action)
            # Same as the runner: the action details carry the episode's MAVEN noise.
            transition = Transition.from_step(obs, state, action.action, step, **action.details)
            logs.update(trainer.update_step(transition, time_step))
            episode.add(transition)
            obs, state = step.obs, step.state
            done = step.done or step.truncated
        logs.update(trainer.update_episode(episode, episode_num, time_step))
    assert "worker/td-loss" in logs
    assert "worker/maven-loss" in logs
    if z_policy_type == "return":
        assert "meta/mean_loss" in logs
    assert all(math.isfinite(value) for value in logs.values())


@pytest.mark.parametrize("noise_size", [1, NOISE_SIZE])
@pytest.mark.parametrize("wrapped", [False, True])
def test_truncated_episode_bootstraps_with_episode_noise(noise_size, wrapped):
    env = env_config(noise_size).make()
    agent = make_maven(noise_size=noise_size, z_policy_type="uniform").make_agent()
    if wrapped:
        agent = AgentWrapper(agent)
    trainer = Mock()
    trainer.update_step.return_value = {}
    trainer.update_episode.return_value = {}
    run = SimpleNamespace(n_steps=1, should_test_at=lambda _: False, logger=Mock())

    episode = _train_episode(env, env, agent, trainer, 0, 0, False, True, run)

    assert episode.is_truncated and not episode.is_done
    assert np.any(episode["maven-noise"][0])
    np.testing.assert_array_equal(
        episode.all_extras[-1][:, -noise_size:], np.broadcast_to(episode["maven-noise"][0], (env.n_agents, noise_size))
    )
    transition = next(episode.transitions())
    np.testing.assert_array_equal(
        transition.next_obs.extras[:, -noise_size:], np.broadcast_to(episode["maven-noise"][0], (env.n_agents, noise_size))
    )


@pytest.mark.parametrize("noise_size", [1, NOISE_SIZE])
def test_test_rollout_preserves_final_episode_noise(noise_size):
    env = env_config(noise_size).make()
    agent = make_maven(noise_size=noise_size, z_policy_type="uniform").make_agent()

    episode, _, actions = seeded_rollout(env, agent, seed=0)

    np.testing.assert_array_equal(
        episode.all_extras[-1][:, -noise_size:], np.broadcast_to(actions[0]["maven-noise"], (env.n_agents, noise_size))
    )


def test_serialization_round_trip():
    trainer = make_maven(mi_loss_coef=0.5)
    restored = Trainer.from_dict(trainer.to_dict())
    assert isinstance(restored, algos.MAVEN)
    assert restored.mi_loss_coef == 0.5
    assert isinstance(restored.worker_trainer, MITrainer)
    assert restored.worker_trainer.mi_loss_coef == 0.5
