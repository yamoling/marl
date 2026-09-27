"""Start MARL Studio."""

import logging
import os
import sys
from pathlib import Path

import dotenv
import typed_argparse as tap

logger = logging.getLogger(__name__)


class Arguments(tap.TypedArgs):
    logdir: Path | None = tap.arg(positional=True, default=None, help="Initial logs root (defaults to MARL_STUDIO_LOGS or project logs)")
    port: int = tap.arg(default=5000)


def main(args: Arguments):
    from studio.backend import run

    run(port=args.port, root=args.logdir)


if __name__ == "__main__":
    dotenv.load_dotenv()
    logging.basicConfig(
        handlers=[logging.FileHandler("studio.log", mode="a"), logging.StreamHandler()],
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
    try:
        tap.Parser(Arguments).bind(main).run()
    except Exception:
        logger.exception("An error occurred while starting MARL Studio with command line '%s'", sys.argv)
