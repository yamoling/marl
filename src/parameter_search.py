import marl
from lle import LLE
import run


def main():
    LEARNING_RATES_ACTOR = [1e-3, 5e-4, 1e-4]
    LEARNING_RATES_CRITIC = [1e-3, 1e-4]
    TRAIN_INTERVALS = [512, 64, 32]
    ENTROPY_COEFFICIENTS = [0.005, 0.01, 0.025]
    GAMMAS = [0.95, 0.99]

    for lr_actor in LEARNING_RATES_ACTOR:
        for lr_critic in LEARNING_RATES_CRITIC:
            for train_interval in TRAIN_INTERVALS:
                for entropy_coefficient in ENTROPY_COEFFICIENTS:
                    for gamma in GAMMAS:
                        env = LLE.level(6).builder().agent_id().time_limit(78).build()
                        nn = marl.nn.model_bank.CNN_ActorCritic.from_env(env)
                        ppo_trainer = marl.training.PPOTrainer(
                            nn,
                            marl.nn.mixers.VDN.from_env(env),
                            gamma=gamma,
                            lr_actor=lr_actor,
                            lr_critic=lr_critic,
                            n_epochs=64,
                            eps_clip=0.2,
                            critic_c1=0.5,
                            exploration_c2=entropy_coefficient,
                            train_interval=train_interval,
                            minibatch_size=min(train_interval, 64),
                            grad_norm_clipping=None,
                        )
                        exp = marl.Experiment.create(
                            env,
                            2_000_000,
                            logdir=f"ppo_{lr_actor}_{lr_critic}_{train_interval}_{entropy_coefficient}_{gamma}",
                            trainer=ppo_trainer,
                        )
                        args = run.Arguments(logdir=exp.logdir, n_runs=8, debug=False, _n_processes=8)
                        run.main(args)


if __name__ == "__main__":
    main()
