import math

from marlenv import Episode, Transition

from marl import algos
from marl.env import LLEConfig
from marl.nn import mixers
from marl.nn.model_bank.qnetworks import QMLP


def test_laies_smoke_on_lle_level_6():
    env_config = LLEConfig(6, obs_type="flattened", state_type="flattened", time_limit=20)
    env = env_config.make()
    qnetwork = QMLP(
        env.n_actions,
        env.n_agents,
        env.observation_shape,
        env.extras_shape,
        hidden_sizes=(32,),
        duelling=False,
    )

    # LLE layered states are flattened channel-first. Laser and gem layers are
    # dynamic environment features, unlike agent positions and static map layers.
    area = env.state_size // (2 * env.n_agents + 4)
    external_state_indices = tuple(range(env.n_agents * area, 2 * env.n_agents * area)) + tuple(
        range((2 * env.n_agents + 2) * area, (2 * env.n_agents + 3) * area)
    )
    trainer = algos.LAIES(
        qnetwork=qnetwork,
        mixer=mixers.QMix.from_env(env_config, embed_size=16, hypernet_embed_size=16),
        external_state_indices=external_state_indices,
        memory_size=64,
        batch_size=4,
        train_interval=(1, "episode"),
        estm_hidden_size=16,
        cdi_samples=2,
    )
    agent = trainer.make_agent()
    obs, state = env.reset()
    episode = Episode.new(obs, state)
    episode_num = 0
    logs: dict[str, float] = {}

    for time_step in range(1, 81):
        action = agent.choose_action(obs)
        step = env.step(action.action)
        transition = Transition.from_step(obs, state, action, step)
        logs.update(trainer.update_step(transition, time_step))
        episode.add(transition)
        obs, state = step.obs, step.state
        if step.done:
            episode_num += 1
            logs.update(trainer.update_episode(episode, episode_num, time_step))
            obs, state = env.reset()
            episode = Episode.new(obs, state)
            agent.new_episode()

    assert {"td-loss", "estm-loss", "idi", "cdi", "intrinsic-reward"} <= logs.keys()
    assert all(math.isfinite(value) for value in logs.values())
    assert logs["estm-loss"] >= 0
    assert logs["idi"] >= 0
    assert logs["cdi"] >= 0
    assert logs["intrinsic-reward"] >= 0
