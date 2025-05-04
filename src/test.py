import marlenv.env_pool
from marl.training import PPO, DQN
from marl.nn import model_bank
from marl import Experiment
from lle import LLE
from marl.models import TransitionMemory
import marl
from marl.models.batch import EpisodeBatch
import marlenv
import torch
from marl.env.agent_distance_shaping import AgentDistanceShaping
from marl.training.multi_trainer import MultiTrainer
from dataclasses import dataclass


def main_ppo():
    env = marlenv.make("HalfCheetah-v2")
    # env = LLE.level(6).obs_type("flattened").pbrs().builder().agent_id().time_limit(78).build()
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

    exp = Experiment.create(env, 2_000_000, trainer=trainer, logdir="debug")
    exp.run(render_tests=True, n_tests=10)


def main_dqn():
    env = LLE.level(6).builder().time_limit(78).build()
    env = AgentDistanceShaping(env)
    # env = marlenv.make("CartPole-v1")
    # env = LLE.level(3).obs_type("flattened").builder().agent_id().time_limit(78).build()
    qnetwork = model_bank.qnetworks.IndependentCNN.from_env(env, mlp_noisy=True)
    # start_index = env.extras_meanings.index("Agent ID-0")
    ir = None
    # ir = ToMIR(qnetwork, n_agents=env.n_agents, is_individual=True)
    # ir = RandomNetworkDistillation.from_env(env)
    # ir = intrinsic_reward.ICM.from_env(env)
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


@dataclass
class ValueTrainer(marl.Trainer):
    def __init__(self, critic: marl.models.nn.Critic, memory: marl.models.EpisodeMemory, gamma: float = 0.99, batch_size: int = 64):
        super().__init__()
        self.critic = critic
        self.gamma = gamma
        self.memory = memory
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

    def _train(self, batch: EpisodeBatch):
        batch.to(self.device)
        values = self.critic.forward(batch.states, batch.states_extras)
        returns = batch.compute_returns(self.gamma)
        loss = torch.nn.functional.mse_loss(values, returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_episode(self, episode: marlenv.Episode, episode_num: int, time_step: int):
        self.memory.add(episode)
        logs = dict[str, float]()
        if self.memory.can_sample(self.batch_size):
            logs["critic-loss"] = self._train(self.memory.sample(self.batch_size))
        return logs

    def value(self, _: marlenv.Observation, state: marlenv.State) -> float:
        data = torch.tensor(state.data, dtype=torch.float32).unsqueeze(0).to(self.device)
        extras = torch.tensor(state.extras, dtype=torch.float32).unsqueeze(0).to(self.device)
        v = self.critic.forward(data, extras).item()
        return float(v)


def main_curriculum():
    envs = []
    for i in range(100):
        filename = f"maps/pool/world-{i}"
        env = LLE.from_file(filename).obs_type("layered").state_type("layered").builder().time_limit(78).build()
        envs.append(env)
    pool = marlenv.env_pool.EnvPool(envs)
    qnetwork = model_bank.qnetworks.IndependentCNN.from_env(env)
    dqn_trainer = marl.training.DQN(
        qnetwork,
        marl.policy.EpsilonGreedy.linear(1, 0.05, 500_000),
        marl.models.TransitionMemory(50_000),
        mixer=marl.nn.mixers.VDN.from_env(pool),
        double_qlearning=True,
        grad_norm_clipping=10,
        gamma=0.95,
    )
    critic_trainer = ValueTrainer(
        critic=model_bank.CNNCritic(env.state_shape, env.state_extra_shape[0]),
        memory=marl.models.EpisodeMemory(5_000),
        gamma=0.95,
        batch_size=64,
    )
    exp = marl.Experiment.create(
        pool,
        n_steps=10_000_000,
        trainer=MultiTrainer(critic_trainer, dqn_trainer),
        agent=dqn_trainer.make_agent(marl.policy.ArgMax()),
        logdir="vdn-value-trainer",
        test_interval=10_000,
    )
    exp.run(n_tests=10)


if __name__ == "__main__":
    main_curriculum()
    # marl.Experiment.load("vdn-value-trainer").run(render_tests=True, n_tests=1)
