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

from os import path


class Arguments(RunArguments):
    logdir: Optional[str] = tap.arg(default=None, help="The experiment directory")
    overwrite: bool = tap.arg(default=False, help="Override the existing experiment directory")
    run: bool = tap.arg(default=False, help="Run the experiment directly after creating it")
    debug: bool = tap.arg(default=False, help="Create the experiment with name 'debug' (overwritten after each run)")
    log_qv: bool = tap.arg(default=False, help="Log qvalues of the experiment")

def create_multiobj_lle(args: Arguments):
    n_steps = 400_000
    test_interval = 5000
    gamma = 0.95
    #env = LLE.level(6).obs_type("layered").state_type("state").build()
    env = LLE.from_file("src/xmarl_extra/lvl6b").obs_type("layered").multi_objective().state_type("state").pbrs().build()
    #env = LLE.level(6).obs_type("layered").multi_objective().state_type("state").build()
    env = marlenv.Builder(env).agent_id().time_limit(78, add_extra=True).build()
    #env = marlenv.Builder(env).time_limit(78, add_extra=True).build()
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
        log_qvalues=args.log_qv,
    )
    logs_path = path.join("logs","")
    if args.logdir is not None:
        if not args.logdir.startswith(logs_path):
            args.logdir = logs_path + args.logdir
    elif args.debug:
        args.logdir = logs_path+"debug"
    else:
        args.logdir = logs_path+env.name
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
        log_qvalues=args.log_qv,
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
