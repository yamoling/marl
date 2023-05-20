import gymnasium as gym
from new_marl.dqn import DQN
import random
import torch
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import polars as pl
from new_marl.policy import EpsilonGreedy



def save_run(name: str, scores: list[list[float]]):
    with open(f"results/{name}.pkl", "wb") as f:
        pickle.dump(scores, f)


def plot():
    files = os.listdir("results")
    for file in files:
        if not file.endswith(".pkl"):
            continue
        with open(f"results/{file}", "rb") as f:
            data = pickle.load(f)
        all_steps = []
        all_scores = []
        for steps, scores in data:
            all_steps += steps
            all_scores += scores
        df = pl.DataFrame({
            "steps": all_steps,
            "scores": all_scores
        })
        df = df.sort("steps")
        df = df.with_columns(df["steps"].apply(lambda x: 500 * round(x / 500)))
        mean = df.groupby("steps").agg(pl.col("scores").mean()).sort("steps")
        std = df.groupby("steps").agg(pl.col("scores").std()).sort("steps")
        
        steps = mean["steps"]
        mean = mean["scores"]
        std = std["scores"]
        
        plt.plot(steps, mean, label=file)
        plt.fill_between(steps, mean - std, mean + std, alpha=0.2)
    plt.legend()
    plt.savefig("results/scores.png")


import multiprocessing as mp   
import rlenv
from new_marl.dqn_2 import DQN2

def worker_dqn(seed: int, return_dict: dict):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    env = gym.make("CartPole-v1")
    policy = EpsilonGreedy(1, 0.05, 10_000)
    dqn = DQN(env, policy, seed)
    return_dict[seed] = dqn.train(1_000)


def worker_dqn2(seed: int, return_dict: dict):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    env = rlenv.make("CartPole-v1")
    env.seed(seed)
    policy = EpsilonGreedy(1, 0.05, 10_000)
    dqn = DQN2(env, policy, seed)
    return_dict[seed] = dqn.train(20_000)

def worker_dqn3(seed: int, return_dict: dict):
    from new_marl.dqn_3 import DQN3
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    env = rlenv.make("CartPole-v1")
    env.seed(seed)
    policy = EpsilonGreedy(1, 0.05, 10_000)
    dqn = DQN3(env, policy, seed)
    return_dict[seed] = dqn.train(20_000)

def worker_ddqn(seed: int, return_dict: dict):
    from new_marl.ddqn import DDQN
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    env = rlenv.make("CartPole-v1")
    env.seed(seed)
    policy = EpsilonGreedy(1, 0.05, 10_000)
    dqn = DDQN(env, policy, seed)
    return_dict[seed] = dqn.train(20_000)

def worker_rdqn(seed: int, return_dict: dict):
    from new_marl.vanilla_rdqn import RDQN
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    env = rlenv.Builder("CartPole-v1").blind(0.3).build()
    env.seed(seed)
    policy = EpsilonGreedy(1, 0.05, 10_000)
    dqn = RDQN(env, policy, seed)
    return_dict[seed] = dqn.train(20_000)



def worker_vdn(seed: int, return_dict: dict):
    from new_marl.mixed_dqn import MixedDQN
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    env = rlenv.make("CartPole-v1")
    env.seed(seed)
    policy = EpsilonGreedy(1, 0.05, 10_000)
    dqn = MixedDQN(env, policy, seed)
    return_dict[seed] = dqn.train(20_000)


def worker_marl(seed: int, return_dict: dict):
    from new_marl.dqn_4 import DQN4
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    env = rlenv.make("CartPole-v1")
    env.seed(seed)
    policy = EpsilonGreedy(1, 0.05, 10_000)
    dqn = DQN4(env, policy, seed)
    return_dict[seed] = dqn.train(20_000)

def worker_recurrent_vdn(seed: int, return_dict: dict):
    from new_marl.recurrent_mixed_dqn import RecurrentMixedDQN
    import marl
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    env = rlenv.Builder("CartPole-v1").build()
    env.seed(seed)
    policy = EpsilonGreedy(1, 0.05, 10_000)
    qnetwork = marl.nn.model_bank.RNNQMix.from_env(env)
    memory = marl.models.EpisodeMemory(5000)
    dqn = RecurrentMixedDQN(
        env, 
        policy, 
        seed, 
        qnetwork, 
        memory, 
        batch_size=64,
        ddqn=True
    )
    return_dict[seed] = dqn.train(1_000_000)


def worker_recurrent_vdn_first_obs(seed: int, return_dict: dict):
    from new_marl.recurrent_mixed_ddqn import RecurrentMixedDQN
    import marl
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    env = rlenv.Builder("smac:3m").build()
    env.seed(seed)
    policy = EpsilonGreedy(1, 0.05, 50_000)
    qnetwork = marl.nn.model_bank.RNNQMix.from_env(env)
    memory = marl.models.EpisodeMemory(5000)
    mixer = marl.qlearning.mixers.VDN(env.n_agents)
    dqn = RecurrentMixedDQN(
        env, 
        policy, 
        seed, 
        qnetwork, 
        memory, 
        mixer=mixer,
        batch_size=64,
        ddqn=True,
        lr=1e-4
    )
    return_dict[seed] = dqn.train(200_000)


def test_marl():
    import marl
    from rlenv.models import EpisodeBuilder, Transition, Episode
    env = rlenv.Builder("CartPole-v1").blind(0.3).build()
    qnetwork = marl.nn.model_bank.RNNQMix.from_env(env)
    mixer = marl.qlearning.mixers.VDN(env.n_agents)
    algo = marl.qlearning.RecurrentMixedDQN(
        qnetwork=qnetwork,
        mixer=mixer,
    )
    def run_episode():
        finished = False
        obs = env.reset()
        score = 0
        episode = EpisodeBuilder()
        episode_length = 0
        algo.before_train_episode(0)
        while not finished:
            action = algo.choose_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            transition = Transition(obs, action, reward, done or truncated, info, next_obs)
            algo.after_train_step(transition)
            episode.add(transition)
            finished = done or truncated
            score += reward
            episode_length += 1
            obs = next_obs
        algo.after_train_episode(0, episode.build())
        return score, episode_length

    def train(n_steps: int):
        i = 0
        scores = []
        steps = []
        durations = []
        import time
        while i < n_steps:
            start = time.time()
            score, episode_length = run_episode()
            durations.append((time.time() - start)/episode_length)
            print(f"Step {i}\tscore: {score}\tavg_score: {np.mean(scores[-50:]):.3f}\tavg_duration: {np.mean(durations[-50:]):.5f}")
            i += episode_length
            scores.append(score)
            steps.append(i)
        return steps, scores
    
    train(20_000)


if __name__ == "__main__":
    worker_recurrent_vdn_first_obs(0, {})
    exit(0)
    all_scores = []
    manager = mp.Manager()
    shared_dict = manager.dict()
    processes: list[mp.Process] = []
    for i in range(10):
        process = mp.Process(target=worker_recurrent_vdn_first_obs, kwargs=dict(seed=i, return_dict=shared_dict))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    print(shared_dict)
    all_scores = shared_dict.values()
    save_run("recurrent_vdn-128-first-obs", all_scores)
    plot()
    