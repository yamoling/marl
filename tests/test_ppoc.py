from dataclasses import dataclass

import numpy as np
import torch
from marlenv import Transition
from torch.distributions import Categorical

from marl.models.batch import TransitionBatch
from marl.models.nn.options import OptionCriticNetwork
from marl.training import PPOC


@dataclass
class _FakeObservation:
    data: np.ndarray
    extras: np.ndarray
    available_actions: np.ndarray


@dataclass
class _FakeState:
    data: np.ndarray
    extras: np.ndarray


class _FakeTransition:
    def __init__(self, *, obs, next_obs, state, next_state, action, reward, done, options):
        self.obs = obs
        self.next_obs = next_obs
        self.state = state
        self.next_state = next_state
        self.action = action
        self.reward = reward
        self.done = done
        self.n_agents = action.shape[0]
        self._details = {"options": options}

    def __getitem__(self, key):
        return self._details[key]


class _ShapeAwareOptionCritic(OptionCriticNetwork):
    def __init__(self, n_options: int, n_actions: int):
        super().__init__(n_options)
        self.n_actions = n_actions
        self.q_scale = torch.nn.Parameter(torch.tensor(1.0))

    def __hash__(self):
        return id(self)

    def compute_q_options(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        base = obs.mean(dim=-1, keepdim=True)
        offsets = torch.arange(self.n_options, device=obs.device, dtype=obs.dtype)
        while offsets.ndim < base.ndim:
            offsets = offsets.unsqueeze(0)
        return self.q_scale * (base + offsets)

    def termination_probability(self, obs: torch.Tensor, extras: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        del extras, options
        return torch.sigmoid(obs.mean(dim=-1))

    def policy(
        self,
        obs: torch.Tensor,
        extras: torch.Tensor,
        available_actions: torch.Tensor,
        options,
    ) -> Categorical:
        del extras
        if isinstance(options, torch.Tensor):
            option_ids = options.to(device=obs.device)
        else:
            option_ids = torch.tensor(options, device=obs.device)
        logits = obs.mean(dim=-1, keepdim=True).repeat_interleave(self.n_actions, dim=-1)
        logits = logits + torch.arange(self.n_actions, device=obs.device, dtype=obs.dtype)
        logits = logits + option_ids.unsqueeze(-1).to(dtype=obs.dtype)
        logits = logits.masked_fill(~available_actions, -torch.inf)
        return Categorical(logits=logits)


def _make_transition_batch(n_steps: int = 4, n_agents: int = 2, n_actions: int = 3, n_options: int = 4):
    transitions = []
    for step in range(n_steps):
        obs = _FakeObservation(
            data=np.full((n_agents, 3), step, dtype=np.float32),
            extras=np.full((n_agents, 1), step + 0.25, dtype=np.float32),
            available_actions=np.ones((n_agents, n_actions), dtype=np.bool_),
        )
        next_obs = _FakeObservation(
            data=np.full((n_agents, 3), step + 1, dtype=np.float32),
            extras=np.full((n_agents, 1), step + 1.25, dtype=np.float32),
            available_actions=np.ones((n_agents, n_actions), dtype=np.bool_),
        )
        state = _FakeState(data=np.full((5,), step, dtype=np.float32), extras=np.full((2,), step, dtype=np.float32))
        next_state = _FakeState(data=np.full((5,), step + 1, dtype=np.float32), extras=np.full((2,), step + 1, dtype=np.float32))
        action = np.array([(step + agent) % n_actions for agent in range(n_agents)], dtype=np.int64)
        options = np.array([(step + agent) % n_options for agent in range(n_agents)], dtype=np.int64)
        transitions.append(
            _FakeTransition(
                obs=obs,
                next_obs=next_obs,
                state=state,
                next_state=next_state,
                action=action,
                reward=float(step + 1),
                done=step == n_steps - 1,
                options=options,
            )
        )
    return TransitionBatch(transitions)


def _make_trainer(n_agents: int = 2, n_options: int = 4, n_actions: int = 3):
    oc = _ShapeAwareOptionCritic(n_options=n_options, n_actions=n_actions)
    return PPOC(
        oc=oc,
        n_agents=n_agents,
        train_interval=4,
        minibatch_size=2,
        n_epochs=2,
        lr=1e-3,
        train_on="transition",
    )


def test_ppoc_for_individual_learners_keeps_multi_agent_shapes():
    batch = _make_transition_batch()
    expanded = batch.for_individual_learners()

    assert expanded.actions.shape == (4, 2)
    assert expanded.rewards.shape == (4, 2)
    assert expanded.dones.shape == (4, 2)
    assert expanded.masks.shape == (4, 2)
    assert expanded["options"].shape == (4, 2)


def test_ppoc_training_tensors_have_expected_shapes():
    trainer = _make_trainer()
    batch = _make_transition_batch().for_individual_learners()

    log_probs, entropies = trainer._compute_policy_terms(batch)
    returns, advantages = trainer._compute_training_data(batch)

    assert log_probs.shape == batch.actions.shape
    assert entropies.shape == batch.actions.shape
    assert returns.shape == batch.rewards.shape
    assert advantages.shape == batch.rewards.shape
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(entropies).all()
    assert torch.isfinite(returns).all()
    assert torch.isfinite(advantages).all()


def test_ppoc_train_returns_scalar_metrics_for_multi_agent_batch():
    trainer = _make_trainer()
    batch = _make_transition_batch()

    logs = trainer.train(batch, step_num=0)

    assert "ppoc/mean_loss" in logs
    assert "ppoc/mean_critic_loss" in logs
    assert "ppoc/mean_actor_loss" in logs
    assert "ppoc/mean_termination_loss" in logs
    assert np.isfinite(logs["ppoc/mean_loss"])
    assert np.isfinite(logs["ppoc/mean_critic_loss"])
    assert np.isfinite(logs["ppoc/mean_actor_loss"])
    assert np.isfinite(logs["ppoc/mean_termination_loss"])
