import torch
import marl
import rlenv
from marl.qlearning.trainer import (
    DQNTrainer, 
    ValueNode, 
    LossNode, 
    TargetNode, 
    NextQValuesNode, 
    QValuesNode,
    MixNode,
    TrainerBuilder
)
from marl.qlearning import VDN
from lle import LLE, ObservationType

def main():
    # env = rlenv.Builder(LLE.from_file("lvl6", ObservationType.FLATTENED)).time_limit(78).build()
    env = rlenv.Builder("CartPole-v1").build()
    env.reset()

    qnetwork = marl.nn.model_bank.MLP.from_env(env)
    qtarget = marl.nn.model_bank.MLP.from_env(env)
    builder = TrainerBuilder(qnetwork)
    builder.mixer(VDN(env.n_agents))
    trainer = builder.build()
    exit(0)
    
    batch: ValueNode[marl.Batch] = ValueNode(None)
    qvalues = QValuesNode(qnetwork, batch)
    qvalues = MixNode(VDN(env.n_agents), qvalues, batch)
    next_qvalues = NextQValuesNode(qtarget, batch)
    next_qvalues = MixNode(VDN(env.n_agents), next_qvalues, batch)
    qtargets = TargetNode(0.99, next_qvalues, batch)
    loss = LossNode(qvalues, qtargets, batch)
    optimizer = torch.optim.Adam(qnetwork.parameters(), lr=1e-3)
    memory = marl.models.TransitionMemory(10_000)
    trainer = DQNTrainer(
        optimizer=optimizer,
        batch_node=batch,
        loss_node=loss,
        train_interval=1,
        trainable_parameters=[qnetwork],
        targets=[qtarget],
        memory=memory
    )

    from marl.qlearning.dqn_nodes import DQN
    dqn = DQN(qnetwork, trainer)
    logger = marl.logging.CSVLogger("logs/test")
    runner = marl.Runner(env, dqn, logger, test_interval=500, n_steps=10_000)
    runner.train(5)


if __name__ == "__main__":
    main()