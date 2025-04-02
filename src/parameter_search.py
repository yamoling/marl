import marl
from lle import LLE
import random
import multiprocessing as mp


def perform_run(logdir: str, seed: int, device_num: int):
    exp = marl.Experiment.load(logdir)
    print(f"Running {logdir} with seed {seed} on device {device_num}")
    exp.run(seed=seed, n_tests=10, quiet=True, device=device_num)


def main():
    LEARNING_RATES_ACTOR = [1e-3, 5e-4, 1e-4]
    LEARNING_RATES_CRITIC = [1e-3, 1e-4]
    TRAIN_INTERVALS = [512, 64, 32]
    ENTROPY_COEFFICIENTS = [0.005, 0.01, 0.025]
    GAMMAS = [0.99, 0.95]

    logdirs = list[tuple[str, int]]()
    N_SEEDS = 10
    for gamma in GAMMAS:
        for lr_actor in LEARNING_RATES_ACTOR:
            for lr_critic in LEARNING_RATES_CRITIC:
                for train_interval in TRAIN_INTERVALS:
                    for entropy_coefficient in ENTROPY_COEFFICIENTS:
                        env = LLE.level(6).pbrs().builder().agent_id().time_limit(78).build()
                        nn = marl.nn.model_bank.CNN_ActorCritic.from_env(env)
                        ppo_trainer = marl.training.PPO(
                            nn,
                            gamma,
                            marl.nn.mixers.VDN.from_env(env),
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
                        for seed in range(N_SEEDS):
                            logdirs.append((exp.logdir, seed))
                        break
    random.shuffle(logdirs)
    params = [(logdir, seed, i % 8) for i, (logdir, seed) in enumerate(logdirs)]
    with mp.Pool(16) as pool:
        pool.starmap(perform_run, params)


if __name__ == "__main__":
    main()
