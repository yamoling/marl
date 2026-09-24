import math

import pytest
import torch
from marlenv import Episode, Transition

from marl import algos, policy
from marl.algos.maven.mutual_information_trainer import MITrainer
from marl.env import LLEConfig
from marl.models import Trainer
from marl.nn.model_bank import MAVENQnetwork

NOISE_SIZE = 3


def env_config():
    return LLEConfig(6, obs_type="flattened", state_type="flattened", time_limit=8, maven_noise_size=NOISE_SIZE)


def make_maven(**kwargs) -> algos.MAVEN:
    env = env_config()
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


def test_serialization_round_trip():
    trainer = make_maven(mi_loss_coef=0.5)
    restored = Trainer.from_dict(trainer.to_dict())
    assert isinstance(restored, algos.MAVEN)
    assert restored.mi_loss_coef == 0.5
    assert isinstance(restored.worker_trainer, MITrainer)
    assert restored.worker_trainer.mi_loss_coef == 0.5
