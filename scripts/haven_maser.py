import marl
from marl.models import TransitionMemory
from marl.nn import mixers
from marl.nn.model_bank import qnetworks


def maser():
    env = marl.env.LLEConfig(6, agent_id=True)
    qnetwork = qnetworks.from_env(env)
    memory = TransitionMemory(50_000)
    trainer = marl.algos.MASER(qnetwork, memory, mixer=mixers.VDN.from_env(env))
    exp = marl.Experiment.create(env, trainer)
    exp.run(16, n_jobs=4, save_weights=False, save_actions=False, limit_torch_threads=None, device_affinity=7)


def haven():
    n_subgoals = 8
    env = marl.env.LLEConfig(6, agent_id=True, extra_padding_size=n_subgoals)
    n_workers = env.n_agents
    k = 5
    n_meta_extras = env.extras_shape[0] - n_subgoals
    n_agent_extras = 0
    meta_qnetwork = qnetworks.QCNN(
        n_subgoals,
        n_workers,
        env.observation_shape,
        (n_meta_extras + n_subgoals,),
        duelling=False,
    )
    worker_qnetwork = qnetworks.QCNN(
        env.n_actions,
        n_workers,
        env.observation_shape,
        (n_meta_extras + n_agent_extras + n_subgoals,),
        duelling=False,
    )
    meta_trainer = marl.algos.DQN(meta_qnetwork, TransitionMemory(50_000 // k), mixer=mixers.VDN())
    worker_trainer = marl.algos.DQN(worker_qnetwork, TransitionMemory(50_000), mixer=mixers.VDN())
    trainer = marl.algos.HAVEN(meta_trainer, worker_trainer, n_workers, n_subgoals, k, n_meta_extras, n_agent_extras)
    exp = marl.Experiment.create(env, trainer, logdir="auto")
    exp.run(16, n_jobs=4, save_weights=False, save_actions=False, limit_torch_threads=None, device_affinity=7)


def main():
    maser()
    haven()


if __name__ == "__main__":
    main()
