"""
Local-only security guarantees of MARL Studio (moved from `src/ui/backend/routes/__init__.py`):
Host allow-list, same-origin mutations, CORS, path confinement below the logs root, PID ownership
before signalling, and SPA static confinement.
"""

import logging
import os
import re
from http import HTTPStatus
from pathlib import Path
from signal import SIGINT

import psutil
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .data.records import PID_FILE, verified_launcher
from .errors import ApiError, bad_request, error_response, forbidden, not_found

logger = logging.getLogger(__name__)

LOCAL_HOST_RE = re.compile(r"(?:localhost|127\.0\.0\.1|\[::1\])(?::[0-9]{1,5})?")
DEV_ORIGIN_RE = re.compile(r"http://(?:localhost|127\.0\.0\.1):517[3-9]")
CORS_ORIGIN_RE = r"^http://(?:localhost|127\.0\.0\.1)(?::(?:5000|517[3-9]))?$"
SAFE_METHODS = ("GET", "HEAD", "OPTIONS")


# ---------------------------------------------------------------- HTTP guards


def is_local_host(host: str) -> bool:
    return LOCAL_HOST_RE.fullmatch(host) is not None


def is_allowed_origin(origin: str | None, host: str) -> bool:
    """Same origin as the Host header, or the local Vite dev server. No Origin header is allowed. @ai-generated"""
    return not origin or origin == f"http://{host}" or DEV_ORIGIN_RE.fullmatch(origin) is not None


async def security_middleware(request: Request, call_next):
    """
    Reject non-local Host headers, cross-origin mutations and cross-site fetches, and turn
    unexpected exceptions into `{error, message}` bodies.

    @ai-generated
    """
    host = request.headers.get("host", "")
    if not is_local_host(host):
        return error_response("forbidden", "Local Host required", HTTPStatus.FORBIDDEN)
    cross_site = not is_allowed_origin(request.headers.get("origin"), host) or request.headers.get("sec-fetch-site") == "cross-site"
    if request.method not in SAFE_METHODS and cross_site:
        return error_response("forbidden", "Cross-origin mutation forbidden", HTTPStatus.FORBIDDEN)
    try:
        return await call_next(request)
    except Exception as exc:
        logger.exception("Unhandled exception while handling %s %s", request.method, request.url)
        return error_response("internal-error", f"{type(exc).__name__}: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)


def install(app: FastAPI):
    """
    Install CORS and the security middleware. The security middleware is outermost, so that even
    CORS preflights with a foreign Host are rejected.

    @ai-generated
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=CORS_ORIGIN_RE,
        allow_credentials=False,
        allow_methods=["GET", "HEAD", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    app.middleware("http")(security_middleware)


# ---------------------------------------------------------------- Paths


def safe_experiment_path(root: Path, experiment_id: str) -> Path:
    """
    Resolve an experiment id (POSIX path relative to the logs root) to a directory strictly below
    the root, following symlinks before the confinement check.
    400 for malformed ids or the root itself, 403 outside the root, 404 if it does not exist.

    @ai-generated
    """
    if not isinstance(experiment_id, str) or not experiment_id or "\x00" in experiment_id or "\\" in experiment_id:
        raise bad_request("Invalid experiment id")
    relative = Path(experiment_id)
    if relative.is_absolute():
        raise bad_request("Experiment ids are relative to the logs root")
    root = root.resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise forbidden("Path is outside the logs root")
    if resolved == root:
        raise bad_request("Expected a directory below the logs root")
    if not resolved.is_dir():
        raise not_found(f"Unknown experiment {experiment_id}", "unknown-experiment")
    return resolved


def safe_new_experiment_path(root: Path, experiment_id: str) -> Path:
    """Like `safe_experiment_path` for a destination that must not exist yet. @ai-generated"""
    if not isinstance(experiment_id, str) or not experiment_id or "\x00" in experiment_id or "\\" in experiment_id:
        raise bad_request("Invalid experiment id")
    relative = Path(experiment_id)
    if relative.is_absolute() or ".." in relative.parts:
        raise bad_request("Experiment ids are relative to the logs root, without '..'")
    root = root.resolve()
    parent = (root / relative).parent.resolve()
    if not parent.is_relative_to(root):
        raise forbidden("Path is outside the logs root")
    return parent / relative.name


def safe_dist_file(dist: Path, path: str) -> Path | None:
    """File of the frontend distribution for `path`, or None (SPA fallback). 403 outside `dist`. @ai-generated"""
    root = dist.resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise forbidden("Path is outside the UI distribution")
    return target if path and target.is_file() else None


# ---------------------------------------------------------------- Processes


def run_process(rundir: Path, expdir: Path, root: Path) -> psutil.Process | None:
    """
    Accept a PID only if it belongs to this experiment's launcher process tree (`start_run.py`).
    A PID file alone is not proof of ownership: PIDs can be reused or written by another process.
    A stale PID file (dead process) is removed. Raises 403 if ownership cannot be verified.

    @ai-generated
    """
    pid_file = rundir / PID_FILE
    if not pid_file.exists() and not pid_file.is_symlink():
        return None
    if pid_file.is_symlink():
        raise forbidden("Unsafe run PID file")
    try:
        fd = os.open(pid_file, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            pid = int(os.read(fd, 64).strip())
        finally:
            os.close(fd)
        if pid <= 0:
            raise ValueError("Non-positive PID")
        try:
            process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            pid_file.unlink(missing_ok=True)
            return None
        if verified_launcher(process, expdir, root.resolve()) is not None:
            return process
    except (ValueError, OSError, psutil.Error):
        pass
    raise forbidden("Cannot verify run process ownership", "pid-unverifiable")


def launcher_of(process: psutil.Process, expdir: Path, root: Path) -> psutil.Process | None:
    """@ai-generated"""
    try:
        return verified_launcher(process, expdir, root.resolve())
    except psutil.Error:
        return None


def stop_verified_run(rundir: Path, expdir: Path, root: Path) -> bool:
    """Send SIGINT only to a run whose PID belongs to the verified launcher tree. @ai-generated"""
    process = run_process(rundir, expdir, root)
    if process is None:
        return False
    try:
        process.send_signal(SIGINT)
    except psutil.NoSuchProcess:
        pass
    (rundir / PID_FILE).unlink(missing_ok=True)
    return True


__all__ = [
    "ApiError",
    "install",
    "launcher_of",
    "run_process",
    "safe_dist_file",
    "safe_experiment_path",
    "safe_new_experiment_path",
    "stop_verified_run",
]
