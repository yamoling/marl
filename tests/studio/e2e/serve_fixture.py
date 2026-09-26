"""Serve Studio against disposable, generated fixture logs for browser tests only."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import build_logs


def main() -> None:
    """Create fixture logs in a temporary directory and serve only that directory. @ai-generated"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5199)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="marl-studio-e2e-") as directory:
        root = Path(directory) / "logs"
        build_logs(root, running=False)
        # Set this before importing the app, which creates a module-level default instance.
        os.environ["MARL_STUDIO_LOGS"] = str(root)
        import uvicorn

        from studio.backend.app import create_app

        uvicorn.run(create_app(root), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
