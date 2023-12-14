import marl
import torch
from marl.models import TransitionMemory
from marl.nn import LinearNN
from rlenv import RLEnv, Transition


def train(env: RLEnv):
    step = 0
    obs = env.reset()
    qnetwork = marl.nn.model_bank.MLP.from_env(env)
    memory = marl.models.TransitionMemory(500)
    qtarget = marl.nn.model_bank.MLP.from_env(env)
    mixer = marl.qlearning.QMix(env.state_shape[0], env.n_agents, embed_size=8)
    target_mixer = marl.qlearning.QMix(env.state_shape[0], env.n_agents, embed_size=8)
    policy = marl.policy.EpsilonGreedy.constant(1.0)
    optimiser = torch.optim.Adam(list(qnetwork.parameters()) + list(mixer.parameters()), lr=0.0001)
    algo = marl.qlearning.DQN(qnetwork, policy, policy)
    while step < 10_000:
        step += 1
        action = algo.choose_action(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        transition = Transition(obs, action, reward, done, info, next_obs, truncated)
        learn(transition, memory, qnetwork, qtarget, mixer, target_mixer, optimiser)
    return algo, mixer


def learn(
    transition: Transition,
    memory: TransitionMemory,
    qnetwork: LinearNN,
    qtarget: LinearNN,
    mixer: marl.qlearning.QMix,
    target_mixer: marl.qlearning.QMix,
    optimiser: torch.optim.Optimizer,
):
    memory.add(transition)
    if len(memory) < 32:
        return
    batch = memory.sample(32)
    qvalues = qnetwork.forward(batch.obs, batch.extras)
    qvalues = qvalues.gather(-1, batch.actions).squeeze(-1)
    qvalues = mixer.forward(qvalues, batch.states)

    qtargets = qtarget.forward(batch.obs_, batch.extras_)
    qtargets[batch.available_actions_ == 0] = -torch.inf
    qtargets = qtargets.max(dim=-1)[0]
    qtargets = batch.rewards + 0.99 * target_mixer.forward(qtargets, batch.states_) * (1 - batch.dones)
    qtargets = torch.detach(qtargets)

    td_error = qvalues - qtargets
    optimiser.zero_grad()
    loss = torch.mean(td_error**2)
    loss.backward()
    optimiser.step()


if __name__ == "__main__":
    from marl.utils.two_steps import TwoSteps, State

    env = TwoSteps()
    dqn, mixer = train(env)

    for state in State:
        if state == State.END:
            continue
        env.force_state(state)
        obs = env.observation()
        qvalues = dqn.compute_qvalues(obs)
        import numpy as np

        payoff_matrix = np.zeros((2, 2))
        for a0 in range(2):
            for a1 in range(2):
                qs = torch.tensor([qvalues[0][a0], qvalues[1][a1]]).unsqueeze(0)
                s = torch.tensor(obs.state, dtype=torch.float32).unsqueeze(0)
                res = mixer.forward(qs, s)
                payoff_matrix[a0, a1] = res
        print(f"Learned Payoff Matrix for {state.name}:\n{payoff_matrix}")
