import marl
import torch
from marl.models import Experiment
from marl.utils.two_steps import TwoSteps, State


if __name__ == "__main__":
    env = TwoSteps()

    qnetwork = marl.nn.model_bank.MLP.from_env(env)
    mixer = marl.qlearning.mixers.QMix(env.state_shape[0], 2, 8)
    # mixer = marl.qlearning.mixers.VDN(2)
    memory = marl.models.TransitionMemory(500)
    device = marl.utils.get_device()
    algo = marl.qlearning.MixedDQN(
        qnetwork=qnetwork,
        batch_size=32,
        gamma=0.99,
        train_policy=marl.policy.EpsilonGreedy.constant(1),
        mixer=mixer,
        device=device,
    )
    experiment = Experiment.create("test", algo, env, 10_000, 10_100)
    runner = experiment.create_runner("csv", "tensorboard")
    runner.train(1)

    algo: marl.qlearning.MixedDQN = runner._algo
    mixer = algo.mixer

    for state in State:
        if state == State.END:
            continue
        env.force_state(state)
        obs = env.observation()
        qvalues = algo.compute_qvalues(obs)
        import numpy as np

        payoff_matrix = np.zeros((2, 2))
        for a0 in range(2):
            for a1 in range(2):
                qs = (
                    torch.tensor([qvalues[0][a0], qvalues[1][a1]])
                    .to(device)
                    .unsqueeze(0)
                )
                s = torch.tensor(obs.state, dtype=torch.float32).to(device).unsqueeze(0)
                res = mixer.forward(qs, s)
                print("Mixed qvalue:", res)
                payoff_matrix[a0, a1] = res
        print(f"Learned Payoff Matrix for {state.name}:\n{payoff_matrix}")
