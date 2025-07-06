import numpy as np
import pathlib
import os
import typed_argparse as tap


class Arguments(tap.TypedArgs):
    logdir: str = tap.arg(positional=True)

def main(args: Arguments):

    a = np.load(args.logdir)['arr_0']

    if len(a.shape) == 1:
        print("Mean accurracy of distiller model: ")
        print(np.mean(a))
    else:
        acc = a[:,:,0]
        loss = a[:,:,1]
        print("Mean accurracy accross epochs of distiller model per timestep: ")
        print(list(np.mean(acc,axis=0)))
        print()
        print("Mean loss accross epochs of distiller model per timestep: ")
        print(list(np.mean(loss,axis=0)))

if __name__ == "__main__":
    tap.Parser(Arguments).bind(main).run()