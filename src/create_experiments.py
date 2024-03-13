import shutil
import marl
import lle
import rlenv
import typed_argparse as tap
from marl.training import DQNTrainer
from marl.training.ppo_trainer import PPOTrainer
from marl.training.qtarget_updater import SoftUpdate, HardUpdate
from marl.utils import ExperimentAlreadyExistsException
from marl.nn import model_bank


class Arguments(tap.TypedArgs):
    run: bool = tap.arg(default=False, help="Run the experiment directly after creating it")
    debug: bool = tap.arg(default=False, help="Create the experiment with name 'debug' (overwritten after each run)")
    n_tests: int = tap.arg(default=0, help="Number of tests to run")


def create_smac(args: Arguments):
    n_steps = 2_000_000
    env = rlenv.adapters.SMAC("3s_vs_5z")
    smac = env._env
    env = rlenv.Builder(env).agent_id().last_action().build()
    qnetwork = marl.nn.model_bank.RNNQMix.from_env(env)
    memory = marl.models.EpisodeMemory(5_000)
    train_policy = marl.policy.EpsilonGreedy.linear(1.0, 0.05, n_steps=50_000)
    test_policy = train_policy
    smac_unit_state_size: int = 4 + smac.shield_bits_ally + smac.unit_type_bits
    trainer = DQNTrainer(
        qnetwork,
        train_policy=train_policy,
        memory=memory,
        double_qlearning=True,
        target_updater=HardUpdate(200),
        lr=5e-4,
        optimiser="adam",
        batch_size=32,
        train_interval=(1, "episode"),
        gamma=0.99,
        mixer=marl.qlearning.mixers.QPlex(
            n_agents=env.n_agents,
            n_actions=env.n_actions,
            state_size=env.state_shape[0],
            adv_hypernet_embed=64,
            n_heads=10,
            weighted_head=True,
        ),
        grad_norm_clipping=10,
    )

    algo = marl.qlearning.RDQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
        test_policy=test_policy,
    )
    if args.debug:
        logdir = "logs/debug"
    else:
        logdir = f"logs/{env.name}"
        if trainer.mixer is not None:
            logdir += f"-{trainer.mixer.name}-validation"
        else:
            logdir += "-iql"
        if trainer.ir_module is not None:
            logdir += f"-{trainer.ir_module.name}"
        if isinstance(trainer.memory, marl.models.PrioritizedMemory):
            logdir += "-PER"
    return marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=5000, n_steps=n_steps)


def create_ppo_lle():
    n_steps = 300_000
    env = lle.LLE.level(2, lle.ObservationType.LAYERED)
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
        c1=1,
        c2=0.01,
    )

    algo = marl.policy_gradient.PPO(ac_network=ac_network)
    logdir = f"logs/{env.name}-TEST-PPO"
    return marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=5000, n_steps=n_steps)


def create_lle(args: Arguments):
    n_steps = 1_000_000
    gamma = 0.95
    env = lle.LLE.level(6, lle.ObservationType.LAYERED, state_type=lle.ObservationType.FLATTENED, multi_objective=True)
    env = rlenv.Builder(env).time_limit(env.width * env.height // 2, add_extra=False).agent_id().build()

    qnetwork = model_bank.CNN.from_env(env, mlp_sizes=(256, 256))
    memory = marl.models.TransitionMemory(50_000)
    train_policy = marl.policy.EpsilonGreedy.linear(
        1.0,
        0.05,
        n_steps=500_000,
    )
    # rnd = marl.intrinsic_reward.RandomNetworkDistillation(
    #     target=model_bank.CNN(env.observation_shape, env.extra_feature_shape[0], (env.reward_size, 512)),
    #     reward_size=env.reward_size,
    #     normalise_rewards=False,
    #     # gamma=gamma,
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
        ir_module=None,
    )

    algo = marl.qlearning.DQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
        test_policy=marl.policy.ArgMax(),
    )

    if args.debug:
        logdir = "logs/debug"
    else:
        logdir = f"logs/{env.name}-multi-objective-256"
        if trainer.mixer is not None:
            logdir += f"-{trainer.mixer.name}"
        else:
            logdir += "-iql"
        if trainer.ir_module is not None:
            logdir += f"-{trainer.ir_module.name}"
        if isinstance(trainer.memory, marl.models.PrioritizedMemory):
            logdir += "-PER"
    return marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=5000, n_steps=n_steps)


def main(args: Arguments):
    try:
        # exp = create_smac(args)
        # exp = create_ppo_lle()
        exp = create_lle(args)
        print(exp.logdir)
        if args.run:
            exp.create_runner(seed=0).to("auto").train(args.n_tests)
    except ExperimentAlreadyExistsException as e:
        response = ""
        response = input(f"Experiment already exists in {e.logdir}. Overwrite? [y/n] ")
        if response.lower() != "y":
            print("Experiment not created.")
            return
        shutil.rmtree(e.logdir)
        return main(args)


if __name__ == "__main__":
    tap.Parser(Arguments).bind(main).run()
