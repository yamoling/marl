"""CLI tests for the Studio launcher."""

import runpy
from pathlib import Path

import pytest
import typed_argparse as tap


@pytest.mark.parametrize("argv, expected", [([], Path("logs")), (["../other-logs"], Path("../other-logs"))])
def test_studio_logdir_argument(argv, expected, monkeypatch):
    """Verify the positional logs root reaches the server. @ai-generated"""
    script = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts" / "serve_studio.py"))
    args = tap.Parser(script["Arguments"]).parse_args(argv)
    calls = []
    monkeypatch.setattr("studio.backend.run", lambda **kwargs: calls.append(kwargs))

    script["main"](args)

    assert calls == [{"port": 5000, "root": expected}]
