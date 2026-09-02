import math

import torch
from marlenv import Transition

from marl import algos
from marl.algos.intrinsic_reward.social_influence import ModelOfOtherAgents, SocialInfluence
from marl.env import LLEConfig
from marl.models.batch import TransitionBatch
from marl.nn import mixers
from marl.nn.model_bank import actor_critics
from marl.utils import Schedule


def make_trainer(env_config: LLEConfig, mixer=None, **kwargs):
    actor, critic = actor_critics.from_env(env_config, recurrent=False)
    trainer = SocialInfluence(
        actor,
        critic,
        mixer,
        moa=ModelOfOtherAgents.from_env(env_config, hidden_size=16),
        train_interval=(8, "step"),
        minibatch_size=4,
        n_epochs=2,
        # Small learning rates and gradient clipping: with the defaults, PPO itself diverges to NaN
        # on such a tiny configuration, which has nothing to do with the influence reward.
        lr_actor=1e-5,
        lr_critic=1e-5,
        grad_norm_clipping=1.0,
        **kwargs,
    )
    # `ILinear` layers allocate their parameters with `torch.empty`: without this, the actor starts
    # from uninitialised garbage. `Experiment.run` does the same before training.
    trainer.randomize()
    return trainer


def collect(env, agent, trainer, n_steps: int):
    """Run the environment and feed the transitions to the trainer, returning the last logs."""
    obs, state = env.reset()
    logs: dict = {}
    for time_step in range(1, n_steps + 1):
        action = agent.choose_action(obs)
        step = env.step(action.action)
        transition = Transition.from_step(obs, state, action, step)
        logs.update(trainer.update_step(transition, time_step))
        obs, state = step.obs, step.state
        if step.done or step.truncated:
            obs, state = env.reset()
            agent.new_episode()
    return logs


def test_order_is_self_first_and_covers_every_agent():
    moa = ModelOfOtherAgents((4,), 0, 3, 5, hidden_size=8)
    order = moa.order
    assert order.tolist() == [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    # Every agent sees itself first and each other agent exactly once.
    for k in range(3):
        assert order[k, 0].item() == k
        assert sorted(order[k].tolist()) == [0, 1, 2]


def test_rolled_actions_put_own_action_first():
    n_agents, n_actions = 3, 4
    moa = ModelOfOtherAgents((4,), 0, n_agents, n_actions, hidden_size=8)
    actions = torch.tensor([[[0, 1, 2]]])  # (time=1, batch=1, n_agents)
    one_hot = torch.nn.functional.one_hot(actions, n_actions).float()
    rolled = moa.rolled_actions(one_hot).view(1, 1, n_agents, n_agents, n_actions)
    for k in range(n_agents):
        # Slot i of agent k holds the one-hot action of agent (k + i) % n_agents
        for i in range(n_agents):
            assert rolled[0, 0, k, i].argmax().item() == actions[0, 0, (k + i) % n_agents].item()


def test_counterfactuals_match_a_factual_forward_pass():
    """The counterfactual for the action that was actually taken must reproduce the MOA output."""
    torch.manual_seed(0)
    n_agents, n_actions = 3, 4
    moa = ModelOfOtherAgents((6,), 2, n_agents, n_actions, hidden_size=8)
    time, batch = 5, 2
    obs = torch.randn(time, batch, n_agents, 6)
    extras = torch.randn(time, batch, n_agents, 2)
    actions = torch.randint(0, n_actions, (time, batch, n_agents))
    one_hot = torch.nn.functional.one_hot(actions, n_actions).float()
    joint = moa.rolled_actions(one_hot)

    logits, previous_hidden = moa.forward_with_history(obs, extras, joint)
    counterfactual = moa.counterfactuals(obs, extras, joint, previous_hidden)
    index = actions.view(1, *actions.shape, 1, 1).expand(1, *counterfactual.shape[1:])
    factual = torch.gather(counterfactual, 0, index).squeeze(0)
    assert torch.allclose(factual, logits, atol=1e-5)


def test_influence_is_zero_when_the_moa_ignores_own_action():
    """
    If the model of other agents does not depend on the influencer's own action, the conditional and
    the marginal distributions coincide and the influence reward must vanish.
    """
    torch.manual_seed(0)
    env_config = LLEConfig(2, obs_type="flattened", time_limit=10)
    trainer = make_trainer(env_config, influence_reward_clip=None)
    # Zero out the weights that read the observing agent's own action (the first n_actions inputs
    # of the action block, which is appended after the encoded observation).
    n_actions = trainer.moa.n_actions
    with torch.no_grad():
        offset = trainer.moa.hidden_size
        trainer.moa.input_layer.weight[:, offset : offset + n_actions] = 0.0

    env = env_config.make()
    agent = trainer.make_agent()
    collect(env, agent, trainer, 32)

    batch = _last_batch(env, agent, trainer)
    reward, influence = trainer._influence_reward(batch)
    assert torch.allclose(influence, torch.zeros_like(influence), atol=1e-6)
    assert torch.allclose(reward, torch.zeros_like(reward), atol=1e-6)


def _last_batch(env, agent, trainer) -> TransitionBatch:
    """Build a small transition batch by rolling out the environment."""
    transitions = []
    obs, state = env.reset()
    while len(transitions) < 8:
        action = agent.choose_action(obs)
        step = env.step(action.action)
        transitions.append(Transition.from_step(obs, state, action, step))
        obs, state = step.obs, step.state
        if step.done or step.truncated:
            obs, state = env.reset()
            agent.new_episode()
    batch = TransitionBatch(transitions)
    return batch.for_individual_learners()  # type: ignore[return-value]


def test_influence_matches_a_naive_reference_implementation():
    """Cross-check the vectorised influence reward against an explicit per-agent loop."""
    torch.manual_seed(1)
    env_config = LLEConfig(2, obs_type="flattened", time_limit=10)
    trainer = make_trainer(env_config, influence_reward_clip=None, influence_weight=Schedule.constant(1.0))
    trainer.moa.randomize()
    env = env_config.make()
    agent = trainer.make_agent()
    batch = _last_batch(env, agent, trainer)

    _, influence = trainer._influence_reward(batch)

    moa = trainer.moa
    obs, extras, actions, joint = trainer._moa_inputs(batch)
    n_agents, n_actions = moa.n_agents, moa.n_actions
    with torch.no_grad():
        _, previous_hidden = moa.forward_with_history(obs, extras, joint)
        counterfactual = torch.softmax(moa.counterfactuals(obs, extras, joint, previous_hidden), dim=-1)
        policy = trainer.actor.policy(batch.obs, batch.extras, available_actions=batch.available_actions)
        probs = trainer._time_major(batch, policy.probs)  # type: ignore[union-attr]

    time, n_batch = obs.shape[0], obs.shape[1]
    expected = torch.zeros(time, n_batch, n_agents)
    for t in range(time):
        for b in range(n_batch):
            for k in range(n_agents):
                total = 0.0
                for j in range(n_agents - 1):
                    conditional = counterfactual[actions[t, b, k], t, b, k, j]
                    marginal = sum(probs[t, b, k, a] * counterfactual[a, t, b, k, j] for a in range(n_actions))
                    total += torch.sum(conditional * (torch.log(conditional) - torch.log(marginal))).item()
                expected[t, b, k] = total
    assert torch.allclose(influence, expected, atol=1e-5)


def test_moa_learns_to_predict_a_deterministic_partner():
    """The MOA loss must drop well below ln(n_actions) when the other agent is fully predictable."""
    torch.manual_seed(0)
    n_agents, n_actions = 2, 4
    moa = ModelOfOtherAgents((3,), 0, n_agents, n_actions, hidden_size=32)
    optimizer = torch.optim.Adam(moa.parameters(), lr=1e-2)
    time, batch = 12, 8
    # Agent 1 always mirrors agent 0's previous action; agent 0 acts at random.
    a0 = torch.randint(0, n_actions, (time, batch))
    a1 = torch.cat((torch.zeros(1, batch, dtype=torch.long), a0[:-1]))
    actions = torch.stack((a0, a1), dim=-1)
    obs = torch.zeros(time, batch, n_agents, 3)
    extras = torch.zeros(time, batch, n_agents, 0)
    joint = moa.rolled_actions(torch.nn.functional.one_hot(actions, n_actions).float())
    targets = actions[:, :, moa.order[:, 1:]][1:]

    for _ in range(300):
        logits, _ = moa.forward_with_history(obs, extras, joint)
        loss = torch.nn.functional.cross_entropy(
            logits[:-1].reshape(-1, n_actions),
            targets.reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Agent 0's next action is unpredictable, agent 1's is not, so a perfect model reaches
    # ln(n_actions) / 2 at most: check we are clearly below the uninformed ln(n_actions).
    assert loss.item() < 0.8 * math.log(n_actions)


def test_smoke_ippo_and_mappo_on_lle():
    env_config = LLEConfig(2, obs_type="flattened", time_limit=10)
    for mixer in (None, mixers.VDN.from_env(env_config)):
        trainer = make_trainer(env_config, mixer)
        env = env_config.make()
        logs = collect(env, trainer.make_agent(), trainer, 64)
        assert {"moa-loss", "influence-weight", "mean-influence", "max-influence"} <= logs.keys()
        assert all(math.isfinite(value) for value in logs.values() if isinstance(value, (int, float)))


def test_serialization_round_trip():
    env_config = LLEConfig(2, obs_type="flattened", time_limit=10)
    trainer = make_trainer(env_config)
    restored = algos.SocialInfluence.from_dict(trainer.to_dict())
    assert isinstance(restored, SocialInfluence)
    assert restored.moa.n_agents == trainer.moa.n_agents
    assert restored.moa.n_actions == trainer.moa.n_actions
    assert restored.moa.obs_shape == trainer.moa.obs_shape


def test_visible_others_matches_the_lle_agent_layers():
    """Visibility is read from the agent-position channels, in the model's rolled order."""
    env_config = LLEConfig(6, obs_type="partial5x5", time_limit=20)
    moa = ModelOfOtherAgents.from_env(env_config, hidden_size=8)
    env = env_config.make()
    obs, _ = env.reset()
    n = env.n_agents
    data = torch.from_numpy(obs.data).unsqueeze(0).unsqueeze(0)  # (time=1, batch=1, n_agents, C, H, W)

    visible = moa.visible_others(data)
    assert visible.shape == (1, 1, n, n - 1)
    # Reference: agent j is visible to agent k iff channel j of k's observation is non-empty.
    present = (obs.data[:, :n].reshape(n, n, -1) != 0).any(-1)
    for k in range(n):
        for i in range(n - 1):
            assert bool(visible[0, 0, k, i]) == bool(present[k, (k + i + 1) % n])


def test_visible_others_is_all_true_under_full_observability():
    env_config = LLEConfig(6, obs_type="layered", time_limit=20)
    moa = ModelOfOtherAgents.from_env(env_config, hidden_size=8)
    env = env_config.make()
    obs, _ = env.reset()
    data = torch.from_numpy(obs.data).unsqueeze(0).unsqueeze(0)
    assert moa.visible_others(data).all()


def test_visibility_masks_the_influence_of_invisible_agents():
    """With `visibility='agent-channels'`, invisible influencees must not contribute any reward."""
    torch.manual_seed(0)
    env_config = LLEConfig(6, obs_type="partial5x5", time_limit=20)
    trainer = make_trainer(env_config, visibility="agent-channels", influence_reward_clip=None)
    baseline = make_trainer(env_config, visibility="all", influence_reward_clip=None)
    baseline.moa.load_state_dict(trainer.moa.state_dict())
    baseline.actor.load_state_dict(trainer.actor.state_dict())

    env = env_config.make()
    batch = _last_batch(env, trainer.make_agent(), trainer)
    _, masked = trainer._influence_reward(batch)
    _, unmasked = baseline._influence_reward(batch)

    obs = trainer._time_major(batch, batch.obs)
    visible = trainer.moa.visible_others(obs)
    assert not visible.all(), "the test needs at least one invisible pair to be meaningful"
    assert torch.all(masked <= unmasked + 1e-6)
    assert not torch.allclose(masked, unmasked)


def test_visibility_requires_spatial_observations():
    env_config = LLEConfig(2, obs_type="flattened", time_limit=10)
    trainer = make_trainer(env_config, visibility="agent-channels")
    env = env_config.make()
    batch = _last_batch(env, trainer.make_agent(), trainer)
    try:
        trainer._influence_reward(batch)
    except ValueError as e:
        assert "height, width" in str(e)
    else:
        raise AssertionError("expected a ValueError for flat observations")


def test_smoke_with_partial_observations_and_visibility():
    env_config = LLEConfig(6, obs_type="partial5x5", time_limit=20)
    trainer = make_trainer(env_config, visibility="agent-channels")
    env = env_config.make()
    logs = collect(env, trainer.make_agent(), trainer, 64)
    assert {"moa-loss", "mean-influence", "max-influence"} <= logs.keys()
    assert all(math.isfinite(v) for v in logs.values() if isinstance(v, (int, float)))
