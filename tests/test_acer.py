"""Tests for the multi-agent ACER trainer (`marl.algos.ACER`)."""

import math
from typing import cast

import torch
from marlenv import Episode, MARLEnv, Transition
from marlenv.catalog import DiscreteMockEnv

from marl import algos
from marl.env import EnvConfig
from marl.models.batch import EpisodeBatch
from marl.nn import mixers
from marl.nn.model_bank import actor_critics
from marl.nn.model_bank.qnetworks import QMLP

N_AGENTS = 2
EPISODE_LENGTH = 6


def make_trainer(with_mixer: bool, **kwargs):
    env_config = EnvConfig.from_any(DiscreteMockEnv(n_agents=N_AGENTS, end_game=EPISODE_LENGTH))
    env = env_config.make()
    actor, _ = actor_critics.from_env(env_config, recurrent=False, actor_kwargs={"mlp_sizes": (16,)})
    critic = QMLP(
        env.n_actions,
        env.n_agents,
        env.observation_shape,
        env.extras_shape,
        hidden_sizes=(16,),
        duelling=False,
    )
    mixer = mixers.QMix.from_env(env_config, embed_size=8, hypernet_embed_size=8) if with_mixer else None
    trainer = algos.ACER(actor, critic, mixer, gamma=0.9, **kwargs)
    return env, trainer


def collect_episode(env: MARLEnv, agent, truncate_at: int | None = None):
    """Play one episode, optionally truncated (instead of terminated) after `truncate_at` steps."""
    obs, state = env.reset()
    agent.new_episode()
    episode = Episode.new(obs, state)
    time_step = 0
    while not episode.is_finished:
        action = agent.choose_action(obs)
        step = env.step(action.action)
        time_step += 1
        if truncate_at is not None and time_step == truncate_at:
            step.truncated = True
        episode.add(Transition.from_step(obs, state, action.action, step, **action.details))
        obs, state = step.obs, step.state
    return episode


class TestRetrace:
    def test_equals_monte_carlo_return_when_on_policy_and_zero_critic(self):
        """
        With unit importance weights and a critic that predicts zero everywhere, the Retrace target
        reduces to the discounted Monte Carlo return of the episode.
        """
        env, trainer = make_trainer(with_mixer=False)
        agent = trainer.make_agent()
        episodes = [collect_episode(env, agent), collect_episode(env, agent, truncate_at=EPISODE_LENGTH - 2)]
        batch = cast(EpisodeBatch, EpisodeBatch(episodes).for_individual_learners())
        zeros = torch.zeros_like(batch.rewards)
        ones = torch.ones_like(batch.rewards)

        q_ret = trainer._retrace(batch, zeros, zeros, ones, zeros)

        for t in reversed(range(batch.masks.shape[0])):
            expected = batch.rewards[t] + trainer.gamma * (q_ret[t + 1] if t + 1 < q_ret.shape[0] else 0)
            expected = expected * batch.masks[t]
            assert torch.allclose(q_ret[t], expected, atol=1e-5)

    def test_bootstraps_on_the_next_value_of_truncated_episodes_only(self):
        """
        An episode that was truncated (time limit) must bootstrap on `V(x_{t+1})` at its last time
        step, while an episode that terminated must not.
        """
        env, trainer = make_trainer(with_mixer=False)
        agent = trainer.make_agent()
        terminated = collect_episode(env, agent)
        truncated = collect_episode(env, agent, truncate_at=EPISODE_LENGTH - 2)
        batch = cast(EpisodeBatch, EpisodeBatch([terminated, truncated]).for_individual_learners())
        zeros = torch.zeros_like(batch.rewards)
        next_values = torch.full_like(batch.rewards, 7.0)

        q_ret = trainer._retrace(batch, zeros, zeros, torch.ones_like(zeros), next_values)

        last_terminated = len(terminated) - 1
        last_truncated = len(truncated) - 1
        assert torch.allclose(q_ret[last_terminated, 0], batch.rewards[last_terminated, 0], atol=1e-5)
        expected = batch.rewards[last_truncated, 1] + trainer.gamma * 7.0
        assert torch.allclose(q_ret[last_truncated, 1], expected, atol=1e-5)
        # Padded time steps are zeroed out and never leak into the recursion.
        assert torch.all(q_ret[batch.masked_indices] == 0.0)


class TestUpdate:
    def test_decentralised_update_changes_the_parameters(self):
        self._test_update(with_mixer=False)

    def test_centralised_update_changes_the_parameters(self):
        self._test_update(with_mixer=True)

    def _test_update(self, with_mixer: bool):
        env, trainer = make_trainer(
            with_mixer,
            train_interval=(2, "episode"),
            batch_size=2,
            replay_start=2,
            replay_ratio=2.0,
            grad_norm_clipping=10.0,
        )
        agent = trainer.make_agent()
        before = [p.clone() for p in trainer.actor.parameters()]
        logs = {}
        for episode_num in range(6):
            episode = collect_episode(env, agent)
            logs = trainer.update_episode(episode, episode_num, episode_num * EPISODE_LENGTH) or logs
        assert {"acer/actor-loss", "acer/critic-loss", "acer/kl-divergence"} <= logs.keys()
        assert all(math.isfinite(value) for value in logs.values())
        assert any(not torch.allclose(p, q) for p, q in zip(before, trainer.actor.parameters()))

    def test_average_policy_follows_the_actor(self):
        """The average policy network is a soft copy of the actor, updated after every gradient step."""
        env, trainer = make_trainer(with_mixer=False, train_interval=(1, "episode"), replay_ratio=0.0)
        trainer.randomize()
        agent = trainer.make_agent()
        # `randomize` re-synchronises the average policy with the actor.
        for avg_param, param in zip(trainer.avg_actor.parameters(), trainer.actor.parameters()):
            assert torch.allclose(avg_param, param)
        avg_before = [p.clone() for p in trainer.avg_actor.parameters()]

        trainer.update_episode(collect_episode(env, agent), 0, EPISODE_LENGTH)

        for avg_before_p, avg_param, param in zip(
            avg_before, trainer.avg_actor.parameters(), trainer.actor.parameters()
        ):
            alpha = trainer.trust_region_decay
            assert torch.allclose(avg_param, alpha * avg_before_p + (1 - alpha) * param, atol=1e-6)
            assert not avg_param.requires_grad


class TestTrustRegion:
    def test_gradient_is_unchanged_when_the_constraint_is_satisfied(self):
        """
        When the KL constraint is not violated, the trust region loss must yield the very same gradient
        as the plain ACER objective (Equation (12) of the paper with a zero correction factor).
        """
        env, trainer = make_trainer(with_mixer=False, trust_region_delta=1e6)
        agent = trainer.make_agent()
        batch = cast(EpisodeBatch, EpisodeBatch([collect_episode(env, agent)]).for_individual_learners())
        probs = trainer._probabilities(trainer.actor, batch)
        objective = torch.sum(torch.log(probs.clamp_min(1e-8)).sum(-1) * batch.masks)
        agent_masks = batch.masks

        loss, kl, factor = trainer._trust_region_loss(batch, probs, objective, agent_masks, batch.n_items)
        (trust_region_grad,) = torch.autograd.grad(loss, probs, retain_graph=True)
        (reference_grad,) = torch.autograd.grad(-objective / batch.n_items, probs)

        assert factor == 0.0
        assert kl >= 0.0
        assert torch.allclose(trust_region_grad, reference_grad, atol=1e-6)


class TestAgent:
    def test_records_the_behaviour_probabilities_while_training(self):
        env, trainer = make_trainer(with_mixer=False)
        agent = trainer.make_agent()
        obs, _ = env.reset()

        action = agent.choose_action(obs)
        assert "action_probabilities" in action.details
        probabilities = action.details["action_probabilities"]
        assert probabilities.shape == (env.n_agents, env.n_actions)

        agent.set_testing()
        assert "action_probabilities" not in agent.choose_action(obs).details
