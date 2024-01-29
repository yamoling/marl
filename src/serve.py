import typed_argparse as tap


class Arguments(tap.TypedArgs):
    port: int = tap.arg(default=5000)
    debug: bool = tap.arg(default=False)


def serve(args: Arguments):
    from ui.backend import run

    run(port=args.port, debug=args.debug)


if __name__ == "__main__":
    tap.Parser(Arguments).bind(serve).run()
