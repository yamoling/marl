"""Focused numerical, gradient, routing, and call-return tests for Option-Critic."""

import json
import random
from collections.abc import Sequence
from unittest.mock import patch

import numpy as np
import pytest
import torch
from marlenv import Transition
from marlenv.catalog import DiscreteMockEnv
from torch.distributions import Categorical

from marl import Experiment
from marl.agents.option_agent import OptionAgent
from marl.algos.option_critic import OptionCritic
from marl.algos.qtarget_updater import HardUpdate
from marl.env import LLEConfig
from marl.models import Action
from marl.models.batch import TransitionBatch
from marl.models.nn import NN, QNetwork
from marl.models.nn.options import OptionCriticNetwork
from marl.nn.mixers.vdn import VDN
from marl.nn.model_bank.actor_critics import CategoricalConvActor
from marl.nn.model_bank.generic import CNN, MLP
from marl.nn.model_bank.options import CNNOptionCritic, OptionTermination
from marl.nn.model_bank.qnetworks import QCNN
from marl.policy import ArgMax


@pytest.fixture(scope="module", autouse=True)
def _single_threaded_torch():
    """Keep short CPU tests cheap and restore the caller's thread setting. @ai-generated"""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def _seeded_rngs():
    """Seed CPU test RNGs independently and restore their caller-owned states. @ai-generated"""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        random.seed(11)
        np.random.seed(11)
        torch.random.default_generator.manual_seed(11)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


class _TableOptions(OptionCriticNetwork):
    """Independent state/option parameters make gradient leakage observable."""

    def __init__(self):
        """Create independent trainable heads for two states and options. @ai-generated"""
        super().__init__((1,), 2)
        self.q = torch.nn.Parameter(torch.tensor([[2.0, 5.0], [8.0, 1.0]]))
        self.beta_logits = torch.nn.Parameter(torch.zeros(2, 2))
        self.action_logits = torch.nn.Parameter(torch.tensor([[[0.0, 1.0, -1.0], [1.0, -1.0, 0.0]], [[-1.0, 0.0, 1.0], [0.0, -1.0, 1.0]]]))

    def compute_q_options(self, obs, extras):
        """@ai-generated"""
        return self.q[obs[..., 0].long()]

    def termination_probability(self, obs, extras, options):
        """@ai-generated"""
        probs = self.beta_logits[obs[..., 0].long()].sigmoid()
        return probs.gather(-1, options.reshape(*probs.shape[:-1], 1)).squeeze(-1)

    def policy(self, obs, extras, available_actions, options: Sequence[int] | torch.Tensor):
        """Route toy logits without depending on the production CNN implementation. @ai-generated"""
        logits = self.action_logits[obs[..., 0].long()]
        indices = torch.as_tensor(options, dtype=torch.long, device=obs.device)
        indices = torch.broadcast_to(indices, obs.shape[:-1])
        logits = logits.gather(-2, indices[..., None, None].expand(*indices.shape, 1, 3)).squeeze(-2)
        return Categorical(logits=logits.masked_fill(~available_actions, -torch.inf))

    __hash__ = object.__hash__


def _batch(size=2, *, terminal=False):
    """Build a real batch with deterministic state indices and per-agent options. @ai-generated"""
    env = DiscreteMockEnv(n_agents=2, n_actions=3, end_game=size)
    obs, state = env.reset()
    transitions = []
    for _ in range(size):
        action = Action(np.array([0, 2]), options=[0, 1])
        step = env.step(action)
        transition = Transition.from_step(obs, state, action, step)
        transition.other["options"] = action["options"]
        transitions.append(transition)
        obs, state = step.obs, step.state
    batch = TransitionBatch(transitions)
    batch.obs = torch.zeros(size, 2, 1)
    batch.next_obs = torch.ones(size, 2, 1)
    batch.extras = torch.zeros(size, 2, 0)
    batch.next_extras = torch.zeros(size, 2, 0)
    batch.available_actions = torch.ones(size, 2, 3, dtype=torch.bool)
    batch.rewards = torch.linspace(1.0, -2.0, size)
    batch.dones = torch.full((size,), terminal, dtype=torch.bool)
    # Isolate losses from TransitionBatch's separate dynamic-field dtype issue.
    batch._cache["options"] = torch.tensor([[0, 1]]).expand(size, -1).clone()
    return batch


def _trainer(*, termination_reg=0.2, entropy_reg=0.15):
    """Use deliberately different online/target tables to detect incorrect bootstrap sources. @ai-generated"""
    trainer = OptionCritic(_TableOptions(), 2, gamma=0.5, termination_reg=termination_reg, entropy_reg=entropy_reg)
    with torch.no_grad():
        trainer.target_oc.q.copy_(torch.tensor([[100.0, -100.0], [4.0, 10.0]]))
        trainer.target_oc.beta_logits[1].fill_(torch.logit(torch.tensor(0.25)))
    return trainer


@pytest.mark.parametrize("beta, expected", [(0.0, [4.0, 10.0]), (1.0, [10.0, 10.0]), (0.25, [5.5, 10.0])])
def test_value_on_arrival_interpolates_continuation_and_best_option(beta, expected):
    """U must use the selected option and termination probability at arrival. @ai-generated"""
    oc = _TableOptions()
    with torch.no_grad():
        oc.q[1].copy_(torch.tensor([4.0, 10.0]))
        oc.beta_logits[1].fill_(torch.logit(torch.tensor(beta)))
    obs = torch.ones(3, 2, 1)
    options = torch.tensor([[[0], [1]]]).expand(3, -1, -1)
    actual = oc.value_on_arrival(obs, torch.empty(3, 2, 0), options)
    assert actual.shape == (3, 2)
    torch.testing.assert_close(actual, torch.tensor(expected).expand(3, -1))


@pytest.mark.parametrize("terminal", [False, True])
def test_critic_uses_discounted_arrival_and_masks_terminal_bootstrap(terminal):
    """Check exact TD targets and ensure only selected online Q-values receive gradients. @ai-generated"""
    trainer = _trainer()
    batch = _batch(terminal=terminal)
    loss = trainer.critic_loss(batch)
    current = torch.tensor([[2.0, 5.0], [2.0, 5.0]])
    target = torch.tensor([[1.0], [-2.0]]).expand(-1, 2).clone()
    if not terminal:
        target += 0.5 * torch.tensor([5.5, 10.0])
    torch.testing.assert_close(loss, ((current - target) ** 2).mean())
    loss.backward()
    expected_grad = torch.zeros_like(trainer.oc.q)
    expected_grad[0] = (2 * (current - target) / current.numel()).sum(0)
    torch.testing.assert_close(trainer.oc.q.grad, expected_grad)
    assert all(parameter.grad is None for parameter in trainer.target_oc.parameters())
    assert trainer.oc.beta_logits.grad is None


@pytest.mark.parametrize("terminal", [False, True])
def test_actor_uses_current_option_baseline_and_negative_entropy_bonus(terminal):
    """Check the numerical policy objective independently of network value helpers. @ai-generated"""
    trainer = _trainer()
    batch = _batch(terminal=terminal)
    policy_loss, _ = trainer.actor_loss(batch)
    logits = trainer.oc.action_logits.detach()[0]
    log_probs = logits.log_softmax(-1)
    entropies = -(log_probs.exp() * log_probs).sum(-1)
    selected_log_probs = log_probs[torch.arange(2), torch.tensor([0, 2])]
    returns = torch.tensor([[1.0], [-2.0]]).expand(-1, 2).clone()
    if not terminal:
        returns += 0.5 * torch.tensor([5.5, 10.0])
    advantages = returns - torch.tensor([2.0, 5.0])
    expected = (-selected_log_probs * advantages - 0.15 * entropies).mean()
    torch.testing.assert_close(policy_loss, expected)
    trainer.entropy_reg = 0.0
    unregularized, _ = trainer.actor_loss(batch)
    torch.testing.assert_close(policy_loss - unregularized, -0.15 * entropies.mean())


@pytest.mark.parametrize("terminal", [False, True])
def test_termination_next_state_gradient_sign_regularizer_and_terminal_zero(terminal):
    """Inferior options terminate more often; regularization discourages gratuitous termination. @ai-generated"""
    trainer = _trainer()
    _, termination_loss = trainer.actor_loss(_batch(terminal=terminal))
    # Target next-state advantages are [-6, 0], unlike either online or current-state values.
    coefficient = torch.tensor([-6.0, 0.0]) + trainer.termination_reg
    expected_loss = torch.tensor(0.0) if terminal else (0.5 * coefficient).mean()
    torch.testing.assert_close(termination_loss, expected_loss)
    termination_loss.backward()
    expected_grad = torch.zeros_like(trainer.oc.beta_logits)
    if not terminal:
        expected_grad[1] = 0.25 * coefficient / 2
        assert expected_grad[1, 0] < 0
        assert expected_grad[1, 1] > 0
    torch.testing.assert_close(trainer.oc.beta_logits.grad, expected_grad)
    assert trainer.oc.q.grad is None
    assert all(parameter.grad is None for parameter in trainer.target_oc.parameters())


@pytest.mark.parametrize("loss_index", [0, 1])
def test_actor_losses_detach_q_baselines_and_bootstrap_targets(loss_index):
    """Policy/termination heads train without leaking gradients into either Q table. @ai-generated"""
    trainer = _trainer()
    losses = trainer.actor_loss(_batch())
    losses[loss_index].backward()
    assert trainer.oc.q.grad is None
    assert all(parameter.grad is None for parameter in trainer.target_oc.parameters())
    head = trainer.oc.action_logits if loss_index == 0 else trainer.oc.beta_logits
    assert head.grad is not None
    assert torch.count_nonzero(head.grad) > 0


@pytest.mark.parametrize("size", [1, 3])
def test_actor_passes_unsqueezed_option_tensor_to_policy(size):
    """Retain batch and agent axes, including the singleton online-update batch. @ai-generated"""
    trainer = _trainer()
    batch = _batch(size)
    with patch.object(trainer.oc, "policy", wraps=trainer.oc.policy) as policy:
        trainer.actor_loss(batch)
    options = policy.call_args.args[3]
    assert isinstance(options, torch.Tensor)
    assert options.shape == (size, 2)
    torch.testing.assert_close(options, batch["options"])


class _RoutingHead(NN):
    """Small observation/extras-sensitive heads isolate policy routing from convolutions."""

    def __init__(self, logits):
        """Create a head with trainable option-specific logits. @ai-generated"""
        super().__init__((len(logits),))
        self.logits = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(self, obs, extras):
        """Distinct sample features reveal accidental routing along the wrong axis. @ai-generated"""
        feature = obs[..., 0, 0, 0, None] + 2 * extras[..., :1]
        return self.logits + feature * torch.arange(self.output_size, device=obs.device)

    __hash__ = object.__hash__


class _RoutingQNetwork(QNetwork):
    """Use the real QNetwork interface and duelling conversion with a cheap raw-output head."""

    def __init__(self, outputs, *, duelling=False):
        """Keep option count distinct from the raw duelling output size. @ai-generated"""
        super().__init__(len(outputs) - int(duelling), 2, (1, 1, 1), (1,), duelling=duelling)
        self.head = _RoutingHead(outputs)

    def forward(self, obs, extras, **kwargs):
        """Return raw network outputs for QNetwork.batch_qvalues to decode. @ai-generated"""
        return self.head(obs, extras)

    __hash__ = object.__hash__


@pytest.mark.parametrize("leading_shape", [(2,), (3, 2), (2, 3, 2)])
def test_cnn_option_critic_decodes_duelling_values_without_counting_value_head_as_option(leading_shape):
    """The value head is not an option; Q equals V plus centred advantages. @ai-generated"""
    qnetwork = _RoutingQNetwork([2.0, 6.0, 10.0], duelling=True)
    oc = CNNOptionCritic(
        torch.nn.ModuleList([_RoutingHead([0.0, 1.0, 2.0]), _RoutingHead([2.0, 1.0, 0.0])]),
        qnetwork,
        _RoutingHead([0.0, 0.0]),
    )
    obs = torch.zeros(*leading_shape, 1, 1, 1)
    extras = torch.zeros(*leading_shape, 1)
    assert qnetwork.output_size == 3
    assert oc.n_options == qnetwork.n_actions == 2
    raw = qnetwork(obs, extras)
    torch.testing.assert_close(raw, torch.tensor([2.0, 6.0, 10.0]).expand(*leading_shape, 3))
    values = oc.compute_q_options(obs, extras)
    assert values.shape == (*leading_shape, 2)
    torch.testing.assert_close(values, torch.tensor([8.0, 12.0]).expand(*leading_shape, 2))
    values[..., 0].sum().backward()
    count = int(np.prod(leading_shape))
    torch.testing.assert_close(qnetwork.head.logits.grad, count * torch.tensor([0.5, -0.5, 1.0]))


def _cnn_options():
    """Construct the production routing wrapper around cheap deterministic heads. @ai-generated"""
    return CNNOptionCritic(
        torch.nn.ModuleList([_RoutingHead([2.0, -1.0, 0.0]), _RoutingHead([-2.0, 3.0, 1.0])]),
        _RoutingQNetwork([4.0, 10.0]),
        _RoutingHead([0.0, 0.0]),
    )


@pytest.mark.parametrize("leading_shape", [(2,), (1, 2), (3, 2), (2, 3, 2)])
@pytest.mark.parametrize("option_format", ["list", "tensor"])
def test_cnn_policy_routes_agents_across_leading_axes_and_masks_actions(leading_shape, option_format):
    """Lists select agent-wise heads; tensors can select a different head for every sample. @ai-generated"""
    oc = _cnn_options()
    count = int(np.prod(leading_shape))
    obs = torch.arange(count, dtype=torch.float32).reshape(*leading_shape, 1, 1, 1) / 10
    extras = torch.arange(count, dtype=torch.float32).reshape(*leading_shape, 1) / 20
    if option_format == "list":
        options = [1, 0]
        selected = torch.tensor(options).expand(leading_shape)
    else:
        selected = (torch.arange(count).reshape(leading_shape) // 2 + torch.tensor([1, 0])) % 2
        options = selected
    available = torch.ones(*leading_shape, 3, dtype=torch.bool)
    available[..., 0, 1] = False
    available[..., 1, 0] = False
    original_mask = available.clone()
    expected_logits = torch.empty(*leading_shape, 3)
    flat_obs = obs.reshape(count, 1, 1, 1)
    flat_extras = extras.reshape(count, 1)
    for index, option in enumerate(selected.flatten().tolist()):
        expected_logits.reshape(count, 3)[index] = oc.policies[option](flat_obs[index], flat_extras[index])
    expected = Categorical(logits=expected_logits.masked_fill(~available, -torch.inf))
    actual = oc.policy(obs, extras, available, options)
    assert actual.batch_shape == torch.Size(leading_shape)
    torch.testing.assert_close(actual.probs, expected.probs)
    assert torch.all(actual.probs[~available] == 0)
    assert torch.isfinite(actual.entropy()).all()
    torch.testing.assert_close(available, original_mask)
    (-actual.log_prob(actual.probs.argmax(-1)).mean()).backward()
    assert all(head.logits.grad is not None for head in oc.policies)
    assert all(torch.count_nonzero(head.logits.grad) > 0 for head in oc.policies)


@pytest.mark.parametrize("terminations, expected_options", [([0.0, 0.0], [0, 1]), ([1.0, 0.0], [1, 1]), ([1.0, 1.0], [1, 0])])
def test_agent_call_return_only_reselects_terminated_options(terminations, expected_options):
    """Non-terminated agents retain their active options even when another option is greedy. @ai-generated"""
    oc = _TableOptions()
    agent = OptionAgent(2, 2, oc, ArgMax())
    agent.options = [0, 1]
    obs = torch.zeros(1, 2, 1)
    extras = torch.empty(1, 2, 0)
    with (
        patch.object(oc, "termination_probability", return_value=torch.tensor([terminations])),
        patch.object(oc, "compute_q_options", return_value=torch.tensor([[[0.0, 5.0], [9.0, 1.0]]])),
        patch("marl.agents.option_agent.np.random.random", return_value=np.full((2,), 0.5)),
    ):
        agent.update_options(obs, extras)
    assert agent.options == expected_options
    assert not agent.force_update_next_option


def test_agent_episode_reset_reselects_once_and_action_metadata_is_a_snapshot():
    """Episode starts select greedy options despite beta=0; later decisions use call-return. @ai-generated"""
    oc = _TableOptions()
    agent = OptionAgent(2, 2, oc, ArgMax())
    agent.options = [0, 1]
    env = DiscreteMockEnv(n_agents=2, n_actions=3)
    observation, _ = env.reset()
    with (
        patch.object(oc, "termination_probability", return_value=torch.zeros(1, 2)),
        patch.object(oc, "compute_q_options", return_value=torch.tensor([[[0.0, 5.0], [9.0, 1.0]]])),
        patch.object(oc, "policy", return_value=Categorical(probs=torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]))) as policy,
    ):
        agent.new_episode()
        action = agent.choose_action(observation)
        assert agent.options == [1, 0]
        assert not agent.force_update_next_option
        assert action["options"] == [1, 0]
        assert policy.call_args.args[3] == [1, 0]
        np.testing.assert_array_equal(action, [0, 2])
        agent.options = [0, 1]
        agent.choose_action(observation)
        assert agent.options == [0, 1]
        assert action["options"] == [1, 0]


@pytest.mark.parametrize("size", [1, 3])
@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("loss_kind", ["actor", "critic"])
def test_losses_accept_uncached_float_options_without_mutating_batch(size, terminal, loss_kind):
    """Local integer casts must preserve integer-option losses and the batch's float metadata. @ai-generated"""
    trainer = _trainer()
    reference = _batch(size, terminal=terminal)
    batch = _batch(size, terminal=terminal)
    batch._cache.clear()
    assert "options" not in batch._cache
    loss_function = trainer.actor_loss if loss_kind == "actor" else trainer.critic_loss
    expected = loss_function(reference)
    actual = loss_function(batch)
    assert batch["options"].dtype == torch.float32
    torch.testing.assert_close(batch["options"], reference["options"].float())
    if loss_kind == "actor":
        for result, target in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, target)
        loss = sum(actual)
    else:
        torch.testing.assert_close(actual, expected)
        loss = actual
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in trainer.oc.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainer.oc.parameters() if parameter.grad is not None)
    assert all(parameter.grad is None for parameter in trainer.target_oc.parameters())


def test_update_step_handles_uncached_options_for_fresh_and_replayed_transitions():
    """Exercise real batch construction, actor optimization, and scheduled replay-critic updates. @ai-generated"""
    trainer = _trainer()
    trainer.batch_size = 2
    trainer.critic_train_interval = 2
    transition = _batch().transitions[0]
    transition.obs.data.fill(0)
    transition.next_obs.data.fill(1)
    assert TransitionBatch([transition])["options"].dtype == torch.float32
    before = {name: parameter.detach().clone() for name, parameter in trainer.oc.named_parameters()}
    with patch.object(trainer, "critic_loss", wraps=trainer.critic_loss) as critic_loss:
        for time_step in range(1, 5):
            logs = trainer.update_step(transition, time_step)
            assert {"policy loss", "termination loss"} <= logs.keys()
            assert ("critic loss" in logs) == (time_step % 2 == 0)
            assert all(np.isfinite(value) for name, value in logs.items() if name.endswith("loss"))
        assert critic_loss.call_count == 2
    for name, parameter in trainer.oc.named_parameters():
        assert torch.isfinite(parameter).all()
        assert not torch.equal(parameter, before[name])
    assert all(parameter.grad is None for parameter in trainer.target_oc.parameters())


@pytest.mark.parametrize("size", [1, 2, 3])
@pytest.mark.parametrize("terminal_mode", ["live", "terminal", "mixed"])
def test_mixed_actor_losses_broadcast_global_advantages_and_detach_targets(size, terminal_mode):
    """Check the implemented VDN actor objectives, agent-axis broadcasting, and masking. @ai-generated"""
    trainer = OptionCritic(_TableOptions(), 2, mixer=VDN(), gamma=0.5, termination_reg=0.2, entropy_reg=0.15)
    with torch.no_grad():
        trainer.target_oc.q.copy_(torch.tensor([[100.0, -100.0], [4.0, 10.0]]))
        trainer.target_oc.beta_logits[1].fill_(torch.logit(torch.tensor(0.25)))
        trainer.oc.beta_logits[1].copy_(torch.logit(torch.tensor([0.2, 0.8])))
    batch = _batch(size)
    states = torch.arange(size) % 2
    batch.obs = states[:, None, None].expand(size, 2, 1).float()
    options = torch.tensor([[0, 1], [1, 0], [0, 0]])[:size]
    batch._cache["options"] = options
    batch.actions = torch.tensor([[0, 2], [1, 0], [2, 1]])[:size]
    batch.rewards = torch.tensor([1.0, -2.0, 3.0])[:size]
    if terminal_mode == "mixed":
        batch.dones = torch.arange(size) % 2 == 1
    else:
        batch.dones = torch.full((size,), terminal_mode == "terminal", dtype=torch.bool)

    # Calculate global values directly, without calling any network or mixer helpers.
    current_table = torch.tensor([[2.0, 5.0], [8.0, 1.0]])
    current_values = current_table[states[:, None], options].sum(-1)
    continued_values = torch.where(options == 0, 4.0, 10.0).sum(-1)
    arrival_values = torch.where(options == 0, 5.5, 10.0).sum(-1)
    not_dones = (~batch.dones).float()
    global_returns = batch.rewards + 0.5 * not_dones * arrival_values
    policy_advantages = (global_returns - current_values)[:, None].expand(size, 2)
    termination_coefficients = (continued_values - 20.0 + 0.2)[:, None].expand(size, 2)
    agent_not_dones = not_dones[:, None].expand(size, 2)

    action_logits = trainer.oc.action_logits.detach().clone().requires_grad_()
    selected_logits = action_logits[states[:, None], options]
    log_probs = selected_logits.log_softmax(-1)
    selected_log_probs = log_probs.gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)
    entropies = -(log_probs.exp() * log_probs).sum(-1)
    expected_policy = (-selected_log_probs * policy_advantages - 0.15 * entropies).mean()
    beta_logits = trainer.oc.beta_logits.detach().clone().requires_grad_()
    next_betas = beta_logits[1, options].sigmoid()
    expected_termination = (next_betas * termination_coefficients * agent_not_dones).mean()
    expected_policy_gradient = torch.autograd.grad(expected_policy, action_logits)[0]
    expected_termination_gradient = torch.autograd.grad(expected_termination, beta_logits)[0]

    policy_loss, termination_loss = trainer.actor_loss(batch)
    assert not batch.individual_rewards
    assert batch.rewards.shape == batch.dones.shape == (size,)
    torch.testing.assert_close(policy_loss, expected_policy.detach())
    torch.testing.assert_close(termination_loss, expected_termination.detach())
    if terminal_mode == "terminal":
        assert termination_loss.item() == 0.0
        assert torch.count_nonzero(expected_termination_gradient) == 0

    for index, (loss, head, expected_gradient) in enumerate(
        (
            (policy_loss, trainer.oc.action_logits, expected_policy_gradient),
            (termination_loss, trainer.oc.beta_logits, expected_termination_gradient),
        )
    ):
        trainer.optim.zero_grad(set_to_none=True)
        loss.backward(retain_graph=index == 0)
        assert head.grad is not None
        assert torch.isfinite(head.grad).all()
        torch.testing.assert_close(head.grad, expected_gradient)
        assert trainer.oc.q.grad is None
        other_head = trainer.oc.beta_logits if index == 0 else trainer.oc.action_logits
        assert other_head.grad is None
        assert all(parameter.grad is None for parameter in trainer.target_oc.parameters())


def test_default_update_schedule_accepts_every_step_not_episode_updates():
    """The online actor runs on all steps, independently of the replay critic interval. @ai-generated"""
    trainer = _trainer()
    for time_step in range(18):
        assert trainer.should_update_at(time_step=time_step)
        assert not trainer.should_update_at(episode_num=time_step)


def test_cnn_option_critic_factory_builds_compatible_lle_heads():
    """Use LLE's real layered observations to validate every factory-created head. @ai-generated"""
    config = LLEConfig(6)
    env = config.make()
    observation, _ = env.reset()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(7)
        oc = CNNOptionCritic.from_env(config, n_options=3)
    assert oc.n_options == 3
    assert len(oc.policies) == 3
    assert all(isinstance(head, CategoricalConvActor) for head in oc.policies)
    assert isinstance(oc.q_options, QCNN)
    assert isinstance(oc.options_termination, OptionTermination)
    assert isinstance(oc.options_termination.cnn, CNN)
    assert isinstance(oc.options_termination.mlp, MLP)
    obs, extras, available = observation.as_tensors(torch.device("cpu"), batch_dim=True, actions=True)
    options = torch.arange(env.n_agents).remainder(3).unsqueeze(0)
    with torch.no_grad():
        q_values = oc.compute_q_options(obs, extras)
        terminations = oc.options_termination(obs, extras)
        selected_terminations = oc.termination_probability(obs, extras, options)
        arrival = oc.value_on_arrival(obs, extras, options.unsqueeze(-1))
        policy = oc.policy(obs, extras, available, options)
    assert q_values.shape == (1, env.n_agents, 3)
    assert terminations.shape == q_values.shape
    assert selected_terminations.shape == arrival.shape == (1, env.n_agents)
    assert torch.isfinite(q_values).all()
    assert torch.isfinite(arrival).all()
    assert torch.all((terminations >= 0) & (terminations <= 1))
    torch.testing.assert_close(selected_terminations, terminations.gather(-1, options.unsqueeze(-1)).squeeze(-1))
    assert policy.probs.shape == (1, env.n_agents, env.n_actions)
    assert torch.all(policy.probs[~available] == 0)
    torch.testing.assert_close(policy.probs.sum(-1), torch.ones(1, env.n_agents))


def _assert_cnn_option_recipe_and_state_structure(original, restored):
    """Check registered head recipes and checkpoint layout without assuming JSON stores weights. @ai-generated"""
    assert isinstance(restored, CNNOptionCritic)
    assert isinstance(restored.policies, torch.nn.ModuleList)
    assert restored.n_options == original.n_options == len(restored.policies)
    assert tuple(restored.output_shape) == tuple(original.output_shape)
    assert json.loads(restored.to_json()) == json.loads(original.to_json())
    assert {name for name, _ in restored.named_children()} == {"policies", "q_options", "options_termination"}
    for original_head, restored_head in zip(original.policies, restored.policies, strict=True):
        assert type(restored_head) is type(original_head) is CategoricalConvActor
        assert tuple(restored_head.output_shape) == tuple(original_head.output_shape)
    assert type(restored.q_options) is type(original.q_options) is QCNN
    assert restored.q_options.n_actions == restored.n_options
    assert tuple(restored.q_options.obs_shape) == tuple(original.q_options.obs_shape)
    assert tuple(restored.q_options.extras_shape) == tuple(original.q_options.extras_shape)
    assert restored.q_options.duelling == original.q_options.duelling
    assert type(restored.options_termination) is type(original.options_termination) is OptionTermination
    assert restored.options_termination.n_options == restored.n_options
    assert tuple(restored.options_termination.output_shape) == (restored.n_options,)
    assert tuple(restored.options_termination.obs_shape) == tuple(original.options_termination.obs_shape)
    assert tuple(restored.options_termination.extras_shape) == tuple(original.options_termination.extras_shape)
    original_state = original.state_dict()
    restored_state = restored.state_dict()
    assert original_state.keys() == restored_state.keys()
    for prefix in [*(f"policies.{index}." for index in range(restored.n_options)), "q_options.", "options_termination."]:
        assert any(key.startswith(prefix) for key in restored_state)
    for key, tensor in original_state.items():
        assert restored_state[key].shape == tensor.shape
        assert restored_state[key].dtype == tensor.dtype


def _assert_option_trainer_recipe_and_registration(original, restored):
    """Verify custom replay/updater recipes and bindings to newly reconstructed parameters. @ai-generated"""
    assert isinstance(restored, OptionCritic)
    for field in ("memory_size", "batch_size", "critic_train_interval", "gamma", "lr", "termination_reg", "entropy_reg"):
        assert getattr(restored, field) == getattr(original, field)
    assert restored.memory.max_size == original.memory.max_size == original.memory_size
    assert len(restored.memory) == 0
    assert tuple(restored.train_interval) == tuple(original.train_interval)
    assert isinstance(restored.q_updater, HardUpdate)
    assert restored.target_updater is restored.q_updater
    assert restored.target_updater.update_period == original.target_updater.update_period
    online = list(restored.oc.parameters())
    target = list(restored.target_oc.parameters())
    assert len(online) > 0
    assert len(restored.target_updater.parameters) == len(online)
    assert len(restored.target_updater.target_parameters) == len(target)
    assert all(bound is parameter for bound, parameter in zip(restored.target_updater.parameters, online, strict=True))
    assert all(bound is parameter for bound, parameter in zip(restored.target_updater.target_parameters, target, strict=True))
    assert {id(parameter) for group in restored.optim.param_groups for parameter in group["params"]} == {
        id(parameter) for parameter in online
    }
    assert not {id(parameter) for parameter in online} & {id(parameter) for parameter in original.oc.parameters()}
    _assert_cnn_option_recipe_and_state_structure(original.oc, restored.oc)


def _option_outputs(oc, inputs):
    """Evaluate Q, termination, arrival, and masked policy outputs for checkpoint comparisons. @ai-generated"""
    obs, extras, available, options = inputs
    with torch.no_grad():
        return (
            oc.compute_q_options(obs, extras),
            oc.termination_probability(obs, extras, options),
            oc.value_on_arrival(obs, extras, options.unsqueeze(-1)),
            oc.policy(obs, extras, available, options).probs,
        )


@pytest.mark.parametrize("loader", [CNNOptionCritic, OptionCriticNetwork, NN])
def test_cnn_option_json_roundtrip_and_checkpoint_restore_identical_values(tmp_path, loader):
    """JSON reconstructs registered heads; the separate checkpoint restores their exact weights. @ai-generated"""
    config = LLEConfig(6)
    original = CNNOptionCritic.from_env(config, n_options=3)
    restored = loader.from_json(original.to_json())
    _assert_cnn_option_recipe_and_state_structure(original, restored)
    assert any(not torch.equal(value, restored.state_dict()[key]) for key, value in original.state_dict().items())
    observation, _ = config.make().reset()
    obs, extras, available = observation.as_tensors(torch.device("cpu"), batch_dim=True, actions=True)
    options = torch.arange(config.n_agents).remainder(3).unsqueeze(0)
    inputs = (obs, extras, available, options)
    expected = _option_outputs(original, inputs)
    checkpoint = tmp_path / "option-model-checkpoint"
    checkpoint.mkdir()
    original.save(checkpoint)
    restored.load(checkpoint)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    for actual, value in zip(_option_outputs(restored, inputs), expected, strict=True):
        torch.testing.assert_close(actual, value, rtol=0, atol=0)


def test_option_critic_json_preserves_custom_memory_and_updater_recipe():
    """Registered tensors must not enter updater JSON or prevent recipe reconstruction. @ai-generated"""
    config = LLEConfig(6)
    trainer = OptionCritic(
        CNNOptionCritic.from_env(config, n_options=3),
        config.n_agents,
        memory_size=23,
        batch_size=3,
        critic_train_interval=2,
        q_updater=HardUpdate(7),
        gamma=0.8,
        lr=3e-4,
        termination_reg=0.2,
        entropy_reg=0.15,
    )
    assert len(trainer.target_updater.parameters) > 0
    recipe = json.loads(trainer.to_json())
    assert recipe["memory_size"] == 23
    assert recipe["q_updater"] == {"update_period": 7, "class-name": "HardUpdate", "name": "HardUpdate"}
    assert "target_updater" not in recipe
    restored = OptionCritic.from_json(trainer.to_json())
    _assert_option_trainer_recipe_and_registration(trainer, restored)
    online = next(restored.oc.parameters())
    target = next(restored.target_oc.parameters())
    initial_target = target.detach().clone()
    with torch.no_grad():
        online.add_(0.5)
    for time_step in range(1, 7):
        restored.target_updater.update(time_step)
        torch.testing.assert_close(target, initial_target, rtol=0, atol=0)
    restored.target_updater.update(7)
    torch.testing.assert_close(target, online, rtol=0, atol=0)


def test_option_experiment_create_load_and_model_checkpoint_roundtrip(tmp_path):
    """Persist recipes and materialize registered heads in a unique temporary experiment directory. @ai-generated"""
    config = LLEConfig(6)
    trainer = OptionCritic(
        CNNOptionCritic.from_env(config, n_options=3),
        config.n_agents,
        memory_size=19,
        batch_size=2,
        critic_train_interval=3,
        q_updater=HardUpdate(11),
    )
    directory = tmp_path / "option-critic-experiment"
    experiment = Experiment.create(config, trainer, logdir=directory, n_steps=17, loggers=("csv",))
    assert experiment.logpath.resolve() == directory.resolve()
    assert experiment.experiment_file.is_file()
    recipe = json.loads(experiment.experiment_file.read_bytes())
    assert recipe["trainer"]["memory_size"] == 19
    assert recipe["trainer"]["q_updater"]["update_period"] == 11
    assert len(recipe["trainer"]["oc"]["policies"]) == 3
    restored = Experiment.load(directory)
    assert isinstance(restored, Experiment)
    assert restored.logpath.resolve() == directory.resolve()
    assert restored.n_steps == 17
    assert tuple(restored.loggers) == ("csv",)
    assert json.loads(restored.env.to_json()) == json.loads(config.to_json())
    assert json.loads(restored.test_env.to_json()) == json.loads(config.to_json())
    _assert_option_trainer_recipe_and_registration(trainer, restored.trainer)
    observation, _ = restored.env.make().reset()
    obs, extras, available = observation.as_tensors(torch.device("cpu"), batch_dim=True, actions=True)
    options = torch.arange(config.n_agents).remainder(3).unsqueeze(0)
    inputs = (obs, extras, available, options)
    expected = _option_outputs(trainer.oc, inputs)
    checkpoint = directory / "model-checkpoint"
    checkpoint.mkdir()
    trainer.oc.save(checkpoint)
    restored.trainer.oc.load(checkpoint)
    for key, value in trainer.oc.state_dict().items():
        torch.testing.assert_close(restored.trainer.oc.state_dict()[key], value, rtol=0, atol=0)
    for actual, value in zip(_option_outputs(restored.trainer.oc, inputs), expected, strict=True):
        torch.testing.assert_close(actual, value, rtol=0, atol=0)


def test_option_trainer_checkpoint_keeps_online_and_target_weights_distinct(tmp_path):
    """Trainer checkpoints must preserve online and target modules without filename collisions. @ai-generated"""
    config = LLEConfig(6)
    trainer = OptionCritic(CNNOptionCritic.from_env(config, n_options=2), config.n_agents, memory_size=13, q_updater=HardUpdate(5))
    with torch.no_grad():
        next(trainer.target_oc.parameters()).add_(0.125)
    expected_online = {key: value.clone() for key, value in trainer.oc.state_dict().items()}
    expected_target = {key: value.clone() for key, value in trainer.target_oc.state_dict().items()}
    assert any(not torch.equal(value, expected_target[key]) for key, value in expected_online.items())
    checkpoint = tmp_path / "option-trainer-checkpoint"
    trainer.save(checkpoint)
    restored = OptionCritic.from_json(trainer.to_json())
    restored.load(checkpoint)
    for network, expected in ((restored.oc, expected_online), (restored.target_oc, expected_target)):
        assert network.state_dict().keys() == expected.keys()
        for key, value in expected.items():
            torch.testing.assert_close(network.state_dict()[key], value, rtol=0, atol=0)


def test_actual_lle_actions_and_six_option_critic_updates_are_finite():
    """Run masked actions and both optimizer paths without substituting network or loss helpers. @ai-generated"""
    config = LLEConfig(6, time_limit=4)
    env = config.make()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(11)
        oc = CNNOptionCritic.from_env(config, n_options=2)
        trainer = OptionCritic(oc, env.n_agents, batch_size=2, critic_train_interval=2)
        agent = trainer.make_agent()
        observation, state = env.reset()
        agent.new_episode()
        heads = (oc.policies, oc.q_options, oc.options_termination)
        before = [[parameter.detach().clone() for parameter in head.parameters()] for head in heads]
        saw_unavailable_action = False
        critic_updates = 0
        for time_step in range(1, 7):
            assert trainer.should_update_at(time_step=time_step)
            available = observation.available_actions
            saw_unavailable_action |= bool(np.any(~available))
            action = agent.choose_action(observation, with_details=True)
            probabilities = action["action_probabilities"]
            assert probabilities.shape == available.shape
            assert np.isfinite(probabilities).all()
            assert np.all(probabilities[~available] == 0)
            np.testing.assert_allclose(probabilities.sum(-1), 1.0, atol=1e-6)
            assert np.all(available[np.arange(env.n_agents), action.action])
            assert len(action["options"]) == env.n_agents
            assert all(0 <= option < oc.n_options for option in action["options"])
            step = env.step(action.action)
            transition = Transition.from_step(observation, state, action.action, step, **action.details)
            assert TransitionBatch([transition])["options"].dtype == torch.float32
            logs = trainer.update_step(transition, time_step)
            assert {"policy loss", "termination loss"} <= logs.keys()
            assert ("critic loss" in logs) == (time_step % 2 == 0)
            assert all(np.isfinite(value) for name, value in logs.items() if name.endswith("loss"))
            critic_updates += "critic loss" in logs
            if step.done or step.truncated:
                observation, state = env.reset()
                agent.new_episode()
            else:
                observation, state = step.obs, step.state
        assert saw_unavailable_action
        assert critic_updates == 3
        for head, original in zip(heads, before, strict=True):
            parameters = list(head.parameters())
            assert all(torch.isfinite(parameter).all() for parameter in parameters)
            assert any(not torch.equal(parameter, initial) for parameter, initial in zip(parameters, original, strict=True))
        assert all(parameter.grad is None for parameter in trainer.target_oc.parameters())


def test_repeated_set_testing_preserves_training_options():
    """A second evaluation-mode call must not replace the saved training option state. @ai-generated"""
    train_policy = ArgMax()
    test_policy = ArgMax()
    agent = OptionAgent(2, 2, _TableOptions(), train_policy, test_policy)
    agent.options = [0, 1]
    agent.set_testing()
    assert agent.is_testing
    assert agent.policy is test_policy
    agent.options = [1, 0]
    agent.set_testing()
    assert agent.options == [1, 0]
    agent.set_training()
    assert agent.is_training
    assert agent.policy is train_policy
    assert agent.options == [0, 1]


@pytest.mark.parametrize("pending_episode_reset", [False, True])
def test_evaluation_preserves_training_episode_reset_flag(pending_episode_reset):
    """Evaluation may consume or set its reset flag without changing the training call-return state. @ai-generated"""
    oc = _TableOptions()
    agent = OptionAgent(2, 2, oc, ArgMax())
    agent.options = [0, 1]
    if pending_episode_reset:
        agent.new_episode()
    agent.set_testing()
    with (
        torch.no_grad(),
        patch.object(oc, "termination_probability", return_value=torch.zeros(1, 2)),
        patch.object(oc, "compute_q_options", return_value=torch.tensor([[[0.0, 5.0], [9.0, 1.0]]])),
        patch("marl.agents.option_agent.np.random.random", return_value=np.full((2,), 0.5)),
    ):
        # Consume the pending reset during evaluation, or create an evaluation-only reset.
        if pending_episode_reset:
            agent.update_options(torch.zeros(1, 2, 1), torch.empty(1, 2, 0))
            assert not agent.force_update_next_option
        else:
            agent.new_episode()
            assert agent.force_update_next_option
        agent.set_testing()
        agent.set_training()
        assert agent.options == [0, 1]
        assert agent.force_update_next_option is pending_episode_reset
        agent.update_options(torch.zeros(1, 2, 1), torch.empty(1, 2, 0))
        assert agent.options == ([1, 0] if pending_episode_reset else [0, 1])
        assert not agent.force_update_next_option
