"""The padding-free and shared-forward paths of DQN must be equivalent to the reference computation."""

import pytest
import torch
from marlenv import Episode, Transition

from marl import algos
from marl.env import LLEConfig
from marl.models import EpisodeMemory
from marl.models.batch import EpisodeBatch
from marl.nn import mixers
from marl.nn.model_bank import qnetworks


def env_config(time_limit: int = 15):
    return LLEConfig(6, obs_type="flattened", state_type="flattened", time_limit=time_limit)


def make_trainer(kind: str, mixer_type=mixers.QMix):
    env = env_config()
    match kind:
        case "mlp":
            qnetwork = qnetworks.QMLP.from_env(env, hidden_sizes=(16,))
        case "noisy":
            qnetwork = qnetworks.QMLP.from_env(env, hidden_sizes=(16,), noisy=True)
        case "rnn":
            qnetwork = qnetworks.QRNN.from_env(env, mlp_head_sizes=(16,), mlp_tail_sizes=(16,))
        case _:
            raise ValueError(kind)
    mixer = mixer_type.from_env(env)
    return algos.DQN(qnetwork, EpisodeMemory(64), mixer=mixer, batch_size=3, train_interval=(1, "episode"))


def padded_batch(trainer) -> EpisodeBatch:
    """Episodes of different lengths so that the batch contains padding. @ai-generated"""
    episodes = []
    for time_limit in (3, 5, 8):
        env = env_config(time_limit).make()
        agent = trainer.make_agent()
        obs, state = env.reset()
        episode = Episode.new(obs, state)
        done = False
        while not done:
            action = agent.choose_action(obs)
            step = env.step(action.action)
            episode.add(Transition.from_step(obs, state, action, step))
            obs, state = step.obs, step.state
            done = step.done or step.truncated
        episodes.append(episode)
    return EpisodeBatch(episodes)


def reference_loss(trainer, batch: EpisodeBatch):
    """The DQN loss as computed before the optimisations: separate forward passes on padded tensors. @ai-generated"""
    all_qvalues = trainer.qnetwork.batch_qvalues(batch.obs, batch.extras, masks=batch.masks)
    qvalues = all_qvalues.gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)
    qvalues = trainer.mixer.forward_batch(qvalues, batch, all_qvalues, batch.actions)
    with torch.no_grad():
        next_qvalues = trainer.qtarget.batch_qvalues(batch.all_obs, batch.all_extras, masks=batch.all_masks)[1:]
        trainer.qnetwork.eval()
        for_index = trainer.qnetwork.batch_qvalues(batch.all_obs, batch.all_extras, masks=batch.all_masks)[1:]
        trainer.qnetwork.train()
        indices = for_index.masked_fill(~batch.next_available_actions, -torch.inf).argmax(-1, keepdim=True)
        next_values = next_qvalues.gather(-1, indices).squeeze(-1)
        next_values = trainer.target_mixer.forward_batch(next_values, batch, next_qvalues, indices.squeeze(-1), is_next=True)
        targets = batch.rewards + trainer.gamma * next_values.masked_fill(batch.dones | batch.masked_indices, 0)
    return trainer._compute_td_loss(qvalues, targets, batch)[0]


def gradients(trainer):
    return [p.grad.clone() for p in trainer.target_updater.parameters if p.grad is not None]


@pytest.mark.parametrize("kind", ["mlp", "rnn", "noisy"])
@pytest.mark.parametrize("mixer_type", [mixers.QMix, mixers.VDN])
def test_dqn_train_matches_the_reference_computation(kind: str, mixer_type):
    torch.manual_seed(0)
    trainer = make_trainer(kind, mixer_type)
    episodes = padded_batch(trainer)._base_episodes
    if kind == "noisy":
        # Noisy layers draw new noise at every forward pass in train mode, so compare in eval mode.
        trainer.qnetwork.eval()
        trainer.qtarget.eval()
        trainer.qnetwork.train = lambda mode=True: trainer.qnetwork  # type: ignore[method-assign]
    state = {k: v.clone() for k, v in trainer.qnetwork.state_dict().items()}

    trainer.optimiser.zero_grad()
    expected_loss = reference_loss(trainer, EpisodeBatch(episodes))
    expected_loss.backward()
    expected_gradients = gradients(trainer)

    trainer.optimiser.zero_grad()
    trainer.optimiser.step = lambda *args, **kwargs: None  # type: ignore[method-assign]
    logs = trainer.train(0, EpisodeBatch(episodes))
    assert all(torch.equal(v, trainer.qnetwork.state_dict()[k]) for k, v in state.items())
    assert logs["td-loss"] == pytest.approx(expected_loss.item(), rel=1e-5)
    actual_gradients = gradients(trainer)
    assert len(actual_gradients) == len(expected_gradients) > 0
    for expected, actual in zip(expected_gradients, actual_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
