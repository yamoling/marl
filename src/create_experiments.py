import marl
import rlenv
from lle import LLE, ObservationType
from marl.training import DQNTrainer
from marl.training.ppo_trainer import PPOTrainer
from marl.training.qtarget_updater import SoftUpdate, HardUpdate


def create_smac():
    n_steps = 3_000_000
    env = rlenv.adapters.SMAC("1c3s5z")
    env = rlenv.Builder(env).agent_id().last_action().build()
    qnetwork = marl.nn.model_bank.RNNQMix.from_env(env)
    memory = marl.models.EpisodeMemory(5_000)
    train_policy = marl.policy.EpsilonGreedy.linear(1.0, 0.05, n_steps=50_000)
    test_policy = train_policy
    trainer = DQNTrainer(
        qnetwork,
        train_policy=train_policy,
        memory=memory,
        double_qlearning=True,
        target_updater=HardUpdate(200),
        lr=5e-4,
        optimiser="rmsprop",
        batch_size=32,
        train_interval=(1, "episode"),
        gamma=0.99,
        # mixer=marl.qlearning.VDN(env.n_agents),
        mixer=marl.qlearning.QMix(env.state_shape[0], env.n_agents),
        grad_norm_clipping=10,
    )

    algo = marl.qlearning.RDQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
        test_policy=test_policy,
    )
    logdir = f"logs/{env.name}"
    if trainer.mixer is not None:
        logdir += f"-{trainer.mixer.name}"
    else:
        logdir += "-iql"
    if trainer.ir_module is not None:
        logdir += f"-{trainer.ir_module.name}"
    if isinstance(trainer.memory, marl.models.PrioritizedMemory):
        logdir += "-PER"
    # logdir = "logs/test"
    return marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=5000, n_steps=n_steps)



def create_ppo_lle():
    n_steps = 1_000_000
    env = LLE.level(2, ObservationType.LAYERED)
    env = rlenv.Builder(env).agent_id().time_limit(78, add_extra=True).build()

    ac_network = marl.nn.model_bank.CNN_ActorCritic.from_env(env)
    memory = marl.models.TransitionMemory(20)

    trainer = PPOTrainer(
        network=ac_network,
        memory=memory,
        gamma=0.99,
        batch_size=5,
        lr_critic=1e-4,
        lr_actor=1e-4,
        optimiser="adam",
        train_every="step",
        update_interval=20,
        clip_eps=0.2,
        c1=0.5,
        c2=0,
    )       

    algo = marl.policy_gradient.PPO(
        ac_network=ac_network
    )
    #logdir = f"logs/{env.name}-TEST_PPO"
    logdir = f"logs/{env.name}-TEST-PPO"
    return marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=1000, n_steps=n_steps)
    

def create_lle():
    n_steps = 1_000_000
    gamma = 0.95
    env = LLE.level(6, ObservationType.LAYERED)
    # env = LLE.from_file("maps/lvl6-gems-everywhere", ObservationType.LAYERED)
    env = rlenv.Builder(env).agent_id().time_limit(78, add_extra=True).build()

    qnetwork = marl.nn.model_bank.CNN.from_env(env)
    memory = marl.models.TransitionMemory(50_000)
    train_policy = marl.policy.EpsilonGreedy.linear(
        1.0,
        0.05,
        n_steps=500_000,
    )
    rnd = marl.intrinsic_reward.RandomNetworkDistillation(
        target=marl.nn.model_bank.CNN(env.observation_shape, env.extra_feature_shape[0], 512),
        normalise_rewards=False,
        # gamma=gamma,
    )
    # memory = marl.models.PrioritizedMemory(
    #     memory=memory,
    #     alpha=0.6,
    #     beta=marl.utils.Schedule.linear(0.4, 1.0, n_steps),
    #     td_error_clipping=5.0,
    # )
    trainer = DQNTrainer(
        qnetwork,
        train_policy=train_policy,
        memory=memory,
        optimiser="adam",
        double_qlearning=True,
        target_updater=SoftUpdate(0.01),
        lr=5e-4,
        batch_size=64,
        train_interval=(5, "step"),
        gamma=gamma,
        mixer=marl.qlearning.VDN(env.n_agents),
        # mixer=marl.qlearning.QMix(env.state_shape[0], env.n_agents),
        grad_norm_clipping=10,
        # ir_module=rnd,
    )

    algo = marl.qlearning.DQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
        test_policy=marl.policy.ArgMax(),
    )

    # logdir = f"logs/{env.name}-lvl6-shaping-vdn"
    logdir = f"logs/{env.name}"
    if trainer.mixer is not None:
        logdir += f"-{trainer.mixer.name}"
    else:
        logdir += "-iql"
    if trainer.ir_module is not None:
        logdir += f"-{trainer.ir_module.name}"
    if isinstance(trainer.memory, marl.models.PrioritizedMemory):
        logdir += "-PER"

    logdir += "-bnaic"

    return marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=5000, n_steps=n_steps)


if __name__ == "__main__":
    # exp = create_smac()
    # exp = create_ppo_lle()
    exp = create_lle()
    print(exp.logdir)
    # exp.create_runner(seed=0).to("auto").train(1)
