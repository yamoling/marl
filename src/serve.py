import typed_argparse as tap


class Arguments(tap.TypedArgs):
    port: int = tap.arg(default=5000)


def main(args: Arguments):
    from ui.backend import run

    run(port=args.port)


if __name__ == "__main__":
    tap.Parser(Arguments).bind(main).run()
