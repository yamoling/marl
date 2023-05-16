import marl
import torch
from marl.models import Experiment
from marl.utils.two_steps import TwoSteps, State


if __name__ == "__main__":
    env = TwoSteps()

    qnetwork = marl.nn.model_bank.MLP.from_env(env)
    #3mixer = marl.qlearning.mixers.QMix(env.state_shape[0], 2, 8)
    mixer = marl.qlearning.mixers.VDN(2)
    memory = marl.models.TransitionMemory(500)
    device = marl.utils.get_device()
    algo = marl.qlearning.MixedDQN(
        qnetwork=qnetwork,
        batch_size=32,
        gamma=0.99,
        train_policy=marl.policy.EpsilonGreedy(1),
        mixer=mixer,
        device=device,
    )
    experiment = Experiment.create("test", algo, env, 10_000, 500)
    runner = experiment.create_runner("csv", "tensorboard")
    runner.train(1)
    
    algo: marl.qlearning.MixedDQN = runner._algo
    mixer = algo.mixer

    for state in State:
        if state == State.END:
            continue
        print(f"State: {state}")
        env.force_state(state)
        obs = env.observation()
        print(obs)
        qvalues = algo.compute_qvalues(obs)
        print(f"Q-values: {qvalues}")
        import numpy as np
        payoff_matrix = np.zeros((2, 2))
        for a0 in range(2):
            for a1 in range(2):
                qs = torch.tensor([qvalues[0][a0], qvalues[1][a1]]).to(device)
                s = torch.tensor(obs.state, dtype=torch.float32).to(device)
                res = mixer.forward(qs, s)
                print("Mixed qvalue:", res)
                payoff_matrix[a0, a1] = res
        print(f"Learned Payoff Matrix for {state}:\n{payoff_matrix}")
