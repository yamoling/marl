import typed_argparse as tap
from lle.solver.cooperation_level import CooperationLevelStr

import marl
from marl.env import LLEPool
from marl.nn.model_bank import qnetworks


class Args(tap.TypedArgs):
    cooperation: CooperationLevelStr = tap.arg(default="asymmetric")
    pool_size: int = tap.arg(default=50)


def main(args: Args):
    env = LLEPool("maps/pool/constructive/INDEPENDENT", args.pool_size)
    test_env = LLEPool("maps/pool/constructive/INDEPENDENT", args.pool_size, offset=500)
    trainer = marl.algos.VDN(qnetworks.from_env(env), gamma=0.95, grad_norm_clipping=10.0)
    exp = marl.Experiment(env, trainer, test_env=test_env)
    exp.run()


if __name__ == "__main__":
    tap.Parser(Args).bind(main).run()
