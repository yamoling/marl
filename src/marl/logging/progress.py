from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import orjson

from .logger import Logger


def _progress_endpoint() -> tuple[str, int]:
    host = os.getenv("MARL_PROGRESS_HOST", "127.0.0.1")
    port = int(os.getenv("MARL_PROGRESS_PORT", "8765"))
    return host, port


def _progress_status(progress: float) -> str:
    return "COMPLETED" if progress >= 1.0 else "RUNNING"


def _progress_server_alive(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def ensure_progress_server():
    host, port = _progress_endpoint()
    if _progress_server_alive(host, port):
        return

    subprocess.Popen(
        [sys.executable, "-m", "ui.backend.progress", "--serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if _progress_server_alive(host, port):
            return
        time.sleep(0.05)


class ProgressPublisher:
    def __init__(self, host: str | None = None, port: int | None = None):
        self.host, self.port = _progress_endpoint() if host is None or port is None else (host, port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._connected = False
        self._connect()

    def _connect(self):
        if self._connected:
            return
        try:
            self._socket.connect((self.host, self.port))
        except OSError:
            ensure_progress_server()
            self._socket.close()
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((self.host, self.port))
        self._socket.sendall(orjson.dumps({"role": "publisher"}) + b"\n")
        self._connected = True

    def publish(self, payload: dict[str, Any]):
        message = orjson.dumps(payload) + b"\n"
        try:
            if not self._connected:
                self._connect()
            self._socket.sendall(message)
        except OSError:
            self._connected = False
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._connect()
            self._socket.sendall(message)

    def close(self):
        try:
            self._socket.close()
        except OSError:
            pass


class ProgressLogger(Logger):
    def __init__(self, logdir: Path, seed: int, n_steps: int, n_tests: int):
        super().__init__(logdir)
        self.seed = seed
        self.n_steps = n_steps
        self.n_tests = n_tests
        self.publisher = ProgressPublisher()

    def _publish(self, time_step: int, kind: str):
        progress = 0.0 if self.n_steps <= 0 else min(time_step / self.n_steps, 1.0)
        payload = {
            "event": "run-progress",
            "logdir": str(self.logdir.parent),
            "run": {
                "rundir": str(self.logdir),
                "seed": self.seed,
                "pid": os.getpid(),
                "progress": progress,
                "status": _progress_status(progress),
                "n_tests": self.n_tests,
            },
            "time_step": time_step,
            "kind": kind,
        }
        self.publisher.publish(payload)

    def log_train(self, data: dict[str, Any], time_step: int):
        self._publish(time_step, "train")

    def log_test_episodes(self, episodes: list, time_step: int, save_actions: bool = True):
        self._publish(time_step, "test")

    def log(self, data: dict[str, Any], time_step: int, prefix: str | None = None):
        return None

    def close(self):
        self.publisher.close()
