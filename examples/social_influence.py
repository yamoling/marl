"""
Social influence as intrinsic motivation (Jaques et al., ICML 2019) on the Laser Learning Environment.

Each agent is rewarded for having a causal influence on the actions of the other agents, measured by
the KL divergence between the other agents' predicted next action given the agent's actual action
and the same prediction marginalised over its counterfactual actions. The predictions come from a
Model of Other Agents (MOA) trained by supervised learning, so no agent ever accesses another
agent's policy.
"""

from marl import Experiment, algos
from marl.env import LLEConfig
from marl.nn import mixers
from marl.nn.model_bank import actor_critics
from marl.utils import Schedule


def short_run():
    env = LLEConfig(6, obs_type="layered")
    actor, critic = actor_critics.from_env(env, recurrent=False)
    trainer = algos.SocialInfluence(
        actor,
        critic,
        mixers.VDN.from_env(env),
        moa=algos.ModelOfOtherAgents.from_env(env),
        # LLE's "layered" observations are fully observable, so every agent always sees every other
        # one. With a partial obs_type (e.g. "partial5x5"), set visibility="agent-channels" to
        # restrict the reward to influencees inside the influencer's field of view, as in the paper.
        visibility="all",
        # The paper ramps up the weight of the influence reward over training (curriculum).
        influence_weight=Schedule.linear(0.0, 0.5, 100_000),
        train_interval=(64, "step"),
        grad_norm_clipping=10,
    )
    exp = Experiment.create(env, trainer, logdir="auto", n_steps=10_000)
    exp.run(test_interval=1_000)


if __name__ == "__main__":
    short_run()
