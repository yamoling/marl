import gymnasium as gym
from dqn_torch import Agent
import numpy as np
import polars as pl


def seed(env: gym.Env, seed_value: int = 0):
    env.reset(seed=seed_value)
    import torch
    import random
    import os
    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)


def run(seed_value: int, max_steps: int):
    env = gym.make("CartPole-v1")
    seed(env, seed_value)
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=env.action_space.n, eps_end=0.01, eps_dec=5e-5, input_dims=env.observation_space.shape, lr=1e-3)
    scores, steps = [], []

    step = 0
    while step < max_steps:
        score = 0
        done = False
        observation = env.reset()[0]
        while step < max_steps and not done:
            step += 1
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            done = done or truncated
            agent.learn()
            observation = observation_
        scores.append(score)
        steps.append(step)

        avg_score = np.mean(scores[-100:])

        print("step ", step, "score %.2f" % score, "average score %.2f" % avg_score, "epsilon %.2f" % agent.epsilon)
    return scores, steps


if __name__ == "__main__":
    num_runs = 10
    max_steps = 40_000
    for n in range(num_runs):
        scores, steps = run(max_steps=max_steps, seed_value=n)
        import os

        os.makedirs(f"run_{n}", exist_ok=True)
        pl.DataFrame({"score": scores, "time_step": steps}).write_csv(f"run_{n}/train.csv")
