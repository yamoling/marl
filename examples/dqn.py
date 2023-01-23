#! /usr/bin/python3
from collections import deque
import torch
import rlenv
from rlenv.models import Transition

if __name__ == "__main__":
    env = rlenv.make_env("CartPole-v1")
    model = rlenv.nn.model_bank.MLP(env.observation_shape, env.extra_feature_shape, (env.n_actions, ))
    optimizer = torch.optim.Adam(model.parameters(), 5e-4)
    qlearning = rlenv.qlearning.DQNBuilder(model, optimizer, 0.99, 50000, 32, 200, False, loss_function=rlenv.nn.loss_functions.mse).build()
    policy = rlenv.policies.DecreasingEpsilonGreedy(env.n_actions, env.n_agents, qlearning, decrease_amount=0.0002)

    scores = deque(maxlen=100)
    for e in range(1000):
        done = False
        obs = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward
            t = Transition(obs, action, reward, done, info, obs_)
            agent.after_step(0, t)
            obs = obs_
        scores.append(score)
        print(f"Episode {e:4d} Score = {sum(scores) / len(scores)} Epsilon = {policy._epsilon}")
