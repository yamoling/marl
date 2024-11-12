from lle import LLE, WorldState, World
import marl
import os
import shutil
import json
from torch.multiprocessing import Pool
import marlenv
import numpy as np
from collections import defaultdict
from marl.other.local_graph import LocalGraphTrainer, LocalGraphBottleneckFinder


def get_bottlenecks_stats(finder: LocalGraphBottleneckFinder[WorldState]):
    """Project the edges on the 2D plane"""
    vertex_bottleneck_scores = defaultdict[tuple[int, int], float](int)
    edge_bottleneck_scores = defaultdict[tuple[WorldState, WorldState], float](int)
    for edge, hit_count in finder.hit_count.items():
        score = finder.predict(edge)
        edge_bottleneck_scores[edge] += score
        start, end = edge
        vertices = start.agents_positions + end.agents_positions
        for vertex in vertices:
            vertex_bottleneck_scores[vertex] += score
    return vertex_bottleneck_scores, edge_bottleneck_scores


def compute_heatmap(world: World, finder: LocalGraphBottleneckFinder[WorldState]):
    """Compute the heatmap of the bottlenecks"""
    vertex_scores, _ = get_bottlenecks_stats(finder)
    heatmap = np.zeros((world.height, world.width))
    for (i, j), score in vertex_scores.items():
        heatmap[i, j] = score
    return heatmap


def run_experiment(exp: marl.Experiment, seed: int):
    print(f"Running {exp.logdir}")
    trainer: LocalGraphTrainer = exp.trainer  # type: ignore
    finder = trainer.local_graph
    world = trainer.world

    exp.run(seed=seed, device="auto")
    heatmap = compute_heatmap(world, finder)
    filename = f"{exp.logdir}/heatmap-{seed}.json"
    with open(filename, "w") as f:
        json.dump(heatmap.tolist(), f)


def run_experiments(experiments: list[marl.Experiment], n_seeds: int):
    with Pool(8) as pool:
        handles = list()
        for seed in range(n_seeds):
            for exp in experiments:
                print(f"Queuing {exp.logdir} - seed {seed}")
                h = pool.apply_async(run_experiment, (exp, seed))
                handles.append(h)

        for h in handles:
            h.get()


# TODO: investigate why it does not work when starting with DQN
def create_experiments(n_steps: int):
    experiments = list[marl.Experiment]()
    for train_dqn in [False]:
        for n_agents in range(1, 5):
            map_name = f"maps/subgraph-{n_agents}agents.toml"
            env = LLE.from_file(map_name).obs_type("layered").single_objective()
            world = env.world
            env = marlenv.Builder(env).agent_id().build()

            finder = LocalGraphBottleneckFinder()
            if train_dqn:
                qnetwork = marl.nn.model_bank.CNN.from_env(env)
                policy = marl.policy.EpsilonGreedy.linear(1.0, 0.05, 100_000)
                test_policy = marl.policy.EpsilonGreedy.constant(1.0)
                algo = marl.algo.DQN(qnetwork, policy, test_policy)
                dqn_trainer = marl.training.DQNTrainer(
                    qnetwork,
                    policy,
                    marl.models.TransitionMemory(10_000),
                    mixer=marl.algo.VDN.from_env(env),
                    lr=1e-4,
                    gamma=0.95,
                    train_interval=(100, "step"),
                )
                trainer = LocalGraphTrainer(finder, world, dqn_trainer)
                trainer = trainer.to(marl.utils.get_device())
            else:
                algo = None
                trainer = LocalGraphTrainer(finder, world, None)
            logdir = "logs/dqn-" if train_dqn else "logs/"
            logdir += f"{n_agents}-agents"
            if os.path.exists(logdir):
                shutil.rmtree(logdir)
            exp = marl.Experiment.create(logdir=logdir, trainer=trainer, algo=algo, n_steps=n_steps, test_interval=0, env=env)
            experiments.append(exp)
    return experiments


if __name__ == "__main__":
    logdirs = create_experiments(1_000_000)
    run_experiments(logdirs, 50)
