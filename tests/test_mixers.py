"""
Tests for marl.nn.mixers: VDN and QMix.

QMix is checked for the property that gives it its name: monotonicity of Q_tot in
each individual agent's Q-value, which holds because every weight of the mixing
network is passed through `AbsLayer` before being applied.
"""

import numpy as np
import pytest
import torch
from marlenv import Episode, Transition
from marlenv.catalog import DiscreteMockEnv

from marl.algos.dqn import DQN
from marl.models import TransitionMemory
from marl.models.batch import EpisodeBatch, TransitionBatch
from marl.nn.mixers.qmix import QMix
from marl.nn.mixers.vdn import VDN
from marl.nn.model_bank import qnetworks


class TestVDN:
    def test_forward_sums_over_agents(self):
        mixer = VDN(n_objectives=1)
        qvalues = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = mixer.forward(qvalues, states=None, states_extras=None)
        assert torch.allclose(out, torch.tensor([6.0, 15.0]))

    def test_agent_dim_is_last_when_single_objective(self):
        assert VDN(n_objectives=1).agent_dim == -1

    def test_agent_dim_is_second_to_last_for_multi_objective(self):
        assert VDN(n_objectives=3).agent_dim == -2

    def test_output_shape_reflects_n_objectives(self):
        assert VDN(n_objectives=1).output_shape == (1,)
        assert VDN(n_objectives=4).output_shape == (4,)

    def test_hash_is_stable_by_identity(self):
        mixer = VDN(n_objectives=1)
        assert hash(mixer) == hash(mixer)


def _make_qmix(n_agents=3, state_size=5, state_extras_size=0, embed_size=8, hypernet_embed_size=8):
    return QMix(
        n_agents=n_agents,
        state_size=state_size,
        state_extras_size=state_extras_size,
        embed_size=embed_size,
        hypernet_embed_size=hypernet_embed_size,
        n_objectives=1,
    )


class TestQMix:
    def test_forward_output_shape_matches_batch_dims(self):
        torch.manual_seed(0)
        n_agents, batch = 3, 7
        mixer = _make_qmix(n_agents=n_agents)
        mixer.randomize()
        qvalues = torch.randn(batch, n_agents)
        states = torch.randn(batch, mixer.state_size)
        states_extras = torch.randn(batch, 0)
        out = mixer.forward(qvalues, states, states_extras)
        assert out.shape == (batch,)

    @pytest.mark.parametrize("batch_dims", [(1,), (1, 1), (1, 2), (2, 1)])
    def test_forward_preserves_singleton_batch_and_time_axes(self, batch_dims):
        mixer = _make_qmix(n_agents=2)
        mixer.randomize()
        qvalues = torch.randn(*batch_dims, 2)
        states = torch.randn(*batch_dims, mixer.state_size)
        states_extras = torch.empty(*batch_dims, 0)

        assert mixer(qvalues, states, states_extras).shape == batch_dims

    @pytest.mark.parametrize("episode", [False, True], ids=["transition", "one_step_episode"])
    def test_dqn_singleton_qmix_targets_and_training(self, episode):
        env = DiscreteMockEnv(n_agents=2, n_actions=3, end_game=1)
        obs, state = env.reset()
        action = env.sample_action()
        transition = Transition.from_step(obs, state, action, env.step(action))
        batch = EpisodeBatch([Episode.from_transitions([transition])]) if episode else TransitionBatch([transition])
        trainer = DQN(
            qnetworks.from_env(env, hidden_sizes=(8,)),
            memory=TransitionMemory(10),
            mixer=QMix.from_env(env, embed_size=8, hypernet_embed_size=8),
        )
        expected_shape = (1, 1) if episode else (1,)

        assert batch.rewards.shape == expected_shape
        with torch.no_grad():
            targets = trainer._compute_qtargets(batch)
        assert targets.shape == expected_shape
        assert torch.isfinite(targets).all()
        assert np.isfinite(trainer.train(0, batch)["td-loss"])

    def test_is_monotonic_in_each_agents_qvalue(self):
        """Increasing a single agent's Q-value (all else fixed) must not decrease Q_tot,
        since all mixing weights are passed through an absolute-value layer."""
        torch.manual_seed(0)
        n_agents, batch = 4, 5
        mixer = _make_qmix(n_agents=n_agents)
        mixer.randomize()
        states = torch.randn(batch, mixer.state_size)
        states_extras = torch.randn(batch, 0)
        base_q = torch.randn(batch, n_agents)

        with torch.no_grad():
            base_out = mixer.forward(base_q, states, states_extras)
            for agent in range(n_agents):
                increased_q = base_q.clone()
                increased_q[:, agent] += 5.0
                increased_out = mixer.forward(increased_q, states, states_extras)
                assert torch.all(increased_out >= base_out - 1e-4)

    def test_from_env_infers_dimensions(self):
        from marlenv.catalog import DiscreteMockEnv

        env = DiscreteMockEnv()
        mixer = QMix.from_env(env, embed_size=4, hypernet_embed_size=4)
        assert mixer.n_agents == env.n_agents
        assert mixer.state_size == env.state_size
        assert mixer.state_extras_size == env.state_extras_size

    def test_input_size_accounts_for_state_extras(self):
        mixer = _make_qmix(state_size=5, state_extras_size=3)
        assert mixer.input_size == 8
