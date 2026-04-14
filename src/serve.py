import typed_argparse as tap
import logging
import os
import dotenv
import sys


class Arguments(tap.TypedArgs):
    port: int = tap.arg(default=5000)


def main(args: Arguments):
    from ui.backend import run

    run(port=args.port)


if __name__ == "__main__":
    dotenv.load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("server.log", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        tap.Parser(Arguments).bind(main).run()
    except Exception as e:
        logging.error(f"An error occurred while starting a run with command line '{sys.argv}'.\nError: {e}", exc_info=True)
