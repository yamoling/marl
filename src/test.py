from marl.training import PPO, DQN
from marl.nn import model_bank
from marl import Experiment
from lle import LLE
from marl.training import intrinsic_reward
from marl.policy import EpsilonGreedy
from marl.models import TransitionMemory
import marl


def main_ppo():
    env = LLE.level(6).obs_type("flattened").pbrs().builder().agent_id().time_limit(78).build()
    # env = marlenv.make("CartPole-v1")
    # env = LLE.level(3).obs_type("flattened").builder().agent_id().time_limit(78).build()
    ac = model_bank.SimpleActorCritic.from_env(env)
    trainer = PPO(
        ac,
        # value_mixer=mixers.VDN(1),
        gamma=0.99,
        critic_c1=0.5,
        exploration_c2=0.01,
        n_epochs=20,
        minibatch_size=50,
        train_interval=200,
        lr_actor=3e-4,
        lr_critic=3e-4,
    )

    exp = Experiment.create(env, 2_000_000, trainer=trainer, logdir="logs/test2")
    exp.run(render_tests=True, n_tests=10)


def main_dqn():
    env = LLE.level(6).builder().time_limit(78).build()
    # env = marlenv.make("CartPole-v1")
    # env = LLE.level(3).obs_type("flattened").builder().agent_id().time_limit(78).build()
    qnetwork = model_bank.qnetworks.IndependentCNN.from_env(env, mlp_noisy=True)
    # start_index = env.extras_meanings.index("Agent ID-0")
    ir = None
    # ir = ToMIR(qnetwork, n_agents=env.n_agents, is_individual=True)
    # ir = RandomNetworkDistillation.from_env(env)
    ir = intrinsic_reward.ICM.from_env(env)
    trainer = DQN(
        qnetwork,
        mixer=marl.nn.mixers.VDN.from_env(env),
        train_policy=marl.policy.ArgMax(),
        memory=TransitionMemory(10_000),
        gamma=0.95,
        double_qlearning=True,
        ir_module=ir,
    )

    exp = Experiment.create(env, 2_000_000, trainer=trainer, test_interval=1000)
    exp.run(render_tests=True, n_tests=1)


if __name__ == "__main__":
    main_dqn()
