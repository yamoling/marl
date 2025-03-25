import shutil
import marl
import marlenv
from typing import Optional
import typed_argparse as tap
from marl.training import DQNTrainer
from marl.training.qtarget_updater import SoftUpdate, HardUpdate
from marl.exceptions import ExperimentAlreadyExistsException

from lle import LLE
from run import Arguments as RunArguments, main as run_experiment


class Arguments(RunArguments):
    logdir: Optional[str] = tap.arg(default=None, help="The experiment directory")
    overwrite: bool = tap.arg(default=False, help="Override the existing experiment directory")
    run: bool = tap.arg(default=False, help="Run the experiment directly after creating it")
    debug: bool = tap.arg(default=False, help="Create the experiment with name 'debug' (overwritten after each run)")


def create_multiobj_lle(args: Arguments):
    n_steps = 1_000_000
    test_interval = 5000
    gamma = 0.95
    env = LLE.level(5).obs_type("layered").state_type("state").multi_objective().build()
    #env = LLE.level(5).obs_type("layered").state_type("state").build()
    env = marlenv.Builder(env).centralised().agent_id().time_limit(78, add_extra=True).build()
    #env = marlenv.Builder(env).centralised().time_limit(78, add_extra=True).build()
    test_env = None

    qnetwork = marl.nn.model_bank.CNN.from_env(env)
    memory = marl.models.TransitionMemory(50_000)
    train_policy = marl.policy.EpsilonGreedy.linear(
        1.0,
        0.05,
        n_steps=500_000,
    )
    mixer = marl.training.mixers.VDN.from_env(env)
    #mixer=marl.training.mixers.QMix.from_env(env)
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
        mixer=mixer,
        grad_norm_clipping=10,
        # ir_module=rnd,
    )

    agent = marl.agents.DQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
        test_policy=marl.policy.ArgMax(),
        is_multi_objective=True,
    )

    if args.logdir is not None:
        if not args.logdir.startswith("logs\\"):
            args.logdir = "logs\\" + args.logdir
    elif args.debug:
        args.logdir = "logs\\debug"
    else:
        args.logdir = f"logs\\{env.name}"
        if trainer.mixer is not None:
            args.logdir += f"-{trainer.mixer.name}"
        else:
            args.logdir += "-iql"
        if trainer.ir_module is not None:
            args.logdir += f"-{trainer.ir_module.name}"
        if isinstance(trainer.memory, marl.models.PrioritizedMemory):
            args.logdir += "-PER"
    return marl.Experiment.create(
        logdir=args.logdir,
        agent=agent,
        trainer=trainer,
        env=env,
        test_interval=test_interval,
        n_steps=n_steps,
        test_env=test_env,
    )

def create_lle(args: Arguments):
    from marl.env.wrappers.randomized_lasers import RandomizedLasers

    n_steps = 1_000_000
    test_interval = 5000
    gamma = 0.95
    lle = RandomizedLasers(
        LLE.level(6)
        # LLE.from_file("maps/lvl6-start-above.toml")
        .obs_type("layered")
        .state_type("state")
        # .pbrs(
        #     1.0,
        #     reward_value=1,
        #     lasers_to_reward=[(4, 0), (6, 12)],
        # )
        .build()
    )
    env = marlenv.Builder(lle).agent_id().time_limit(78, add_extra=True).build()

    test_env = None

    qnetwork = marl.nn.model_bank.CNN.from_env(env)
    memory = marl.models.TransitionMemory(50_000)
    # memory = marl.models.PrioritizedMemory(memory, env.is_multi_objective, alpha=0.6, beta=Schedule.linear(0.4, 1.0, n_steps))
    train_policy = marl.policy.EpsilonGreedy.linear(
        1.0,
        0.05,
        n_steps=200_000,
    )
    mixer = marl.training.VDN.from_env(env)
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
        mixer=mixer,
        grad_norm_clipping=10,
        ir_module=None,
    )

    algo = marl.agents.DQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
        test_policy=marl.policy.ArgMax(),
    )

    if args.logdir is not None:
        if not args.logdir.startswith("logs\\"):
            args.logdir = "logs\\" + args.logdir
    elif args.debug:
        args.logdir = "logs\\debug"
    else:
        args.logdir = f"logs\\{env.name}"
        if trainer.mixer is not None:
            args.logdir += f"-{trainer.mixer.name}"
        else:
            args.logdir += "-iql"
        if trainer.ir_module is not None:
            args.logdir += f"-{trainer.ir_module.name}"
        if isinstance(trainer.memory, marl.models.PrioritizedMemory):
            args.logdir += "-PER"
    print(type(env))
    return marl.Experiment.create(
        logdir=args.logdir,
        agent=algo,
        trainer=trainer,
        env=env,
        test_interval=test_interval,
        n_steps=n_steps,
        test_env=test_env,
    )


def main(args: Arguments):
    try:
        exp = create_multiobj_lle(args)
        #exp = create_lle(args)
        print(exp.logdir)
        shutil.copyfile(__file__, exp.logdir + "/tmp.py")
        if args.run:
            args.logdir = exp.logdir
            run_experiment(args)
            # exp.create_runner(seed=0).to("auto").train(args.n_tests)
    except ExperimentAlreadyExistsException as e:
        if not args.overwrite:
            response = ""
            response = input(f"Experiment already exists in {e.logdir}. Overwrite? [y/n] ")
            if response.lower() != "y":
                print("Experiment not created.")
                return
        shutil.rmtree(e.logdir)
        return main(args)


if __name__ == "__main__":
    tap.Parser(Arguments).bind(main).run()
