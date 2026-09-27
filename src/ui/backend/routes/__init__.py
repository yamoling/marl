import logging
import os
import re
from http import HTTPStatus
from pathlib import Path
from signal import SIGINT

import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from ..server_state import ServerState

state = ServerState()

dist_dir = Path(__file__).resolve().parents[2] / "dist"
if not (dist_dir / "index.html").is_file():
    raise RuntimeError("Could not find front end files to serve ! Make sure you have built them (cf: readme).")


app = FastAPI()
logger = logging.getLogger(__name__)


def error_response(exc: Exception, status_code: int | None = None) -> JSONResponse:
    """Format unexpected API errors without changing their status mapping.

    @ai-edited
    """
    if status_code is None:
        match exc:
            case FileNotFoundError():
                status_code = HTTPStatus.NOT_FOUND
            case _:
                status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    return JSONResponse(
        status_code=status_code,
        content={"error": type(exc).__name__, "message": str(exc)},
    )


def safe_log_path(value: str, *, must_exist: bool = True) -> str:
    """Resolve a client path inside the configured logs root independently of cwd.

    @ai-generated
    """
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise HTTPException(HTTPStatus.BAD_REQUEST, "Invalid log path")
    root = Path(state.logdir).resolve()
    path = Path(value)
    if not path.is_absolute():
        # Existing clients send logs/name; also accept root-relative names.
        parts = path.parts
        root_parts = Path(state.logdir).parts
        if parts[: len(root_parts)] == root_parts:
            path = Path.cwd() / path
        elif parts[0] == root.name:
            path = root / Path(*parts[1:])
        else:
            path = root / path
    resolved = path.resolve()
    if resolved == root:
        raise HTTPException(HTTPStatus.BAD_REQUEST, "Expected a directory below the logs root")
    if not resolved.is_relative_to(root):
        raise HTTPException(HTTPStatus.FORBIDDEN, "Path is outside the logs root")
    if must_exist and not resolved.is_dir():
        raise HTTPException(HTTPStatus.NOT_FOUND, "Log directory not found")
    return str(resolved)


def safe_experiment(logdir: str, *, full: bool = False):
    """Load an experiment only when its serialized path names the requested directory.

    The in-memory path is made absolute before any model method walks runs or writes files.
    @ai-generated
    """
    logdir = safe_log_path(logdir)
    exp = state.get_experiment(logdir, full=True) if full else state.get_experiment(logdir)
    if not isinstance(exp.logdir, str) or safe_log_path(exp.logdir, must_exist=False) != logdir:
        raise HTTPException(HTTPStatus.FORBIDDEN, "Experiment metadata points to another directory")
    exp.logdir = logdir
    return exp


def safe_run(run, logdir: str) -> str:
    """Validate and normalize a run's serialized path before using model methods.

    @ai-generated
    """
    if not isinstance(run.rundir, str):
        raise HTTPException(HTTPStatus.FORBIDDEN, "Invalid run metadata")
    rundir = safe_log_path(run.rundir, must_exist=False)
    if Path(rundir).parent != Path(logdir):
        raise HTTPException(HTTPStatus.FORBIDDEN, "Run metadata points outside the experiment")
    if not Path(rundir).is_dir():
        raise HTTPException(HTTPStatus.NOT_FOUND, "Run directory not found")
    run.rundir = rundir
    return rundir


def verified_launcher(process, logdir: Path):
    """Find the launcher ancestor whose command names the validated experiment.

    @ai-generated
    """
    relative = str(logdir.relative_to(Path(state.logdir).resolve().parent))
    for ancestor in (process, *process.parents()):
        command = ancestor.cmdline()
        if any(Path(arg).name == "start_run.py" for arg in command) and any(arg in (str(logdir), relative) for arg in command):
            return ancestor
    return None


def safe_run_process(run):
    """Accept a PID only if it belongs to this experiment's launcher process tree.

    A PID file alone is not proof of ownership: PIDs can be reused or written by
    another process. Do not signal a process if its ancestry cannot be checked.
    @ai-generated
    """
    pid_file = Path(run.rundir) / "pid"
    if not pid_file.exists() and not pid_file.is_symlink():
        return None
    if pid_file.is_symlink():
        raise HTTPException(HTTPStatus.FORBIDDEN, "Unsafe run PID file")
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
        if verified_launcher(process, Path(run.rundir).parent) is not None:
            return process
    except (ValueError, OSError, psutil.Error):
        pass
    raise HTTPException(HTTPStatus.FORBIDDEN, "Cannot verify run process ownership")


def stop_verified_run(run) -> bool:
    """Signal only a run whose PID belongs to the validated launcher tree.

    @ai-generated
    """
    process = safe_run_process(run)
    if process is None:
        return False
    try:
        process.send_signal(SIGINT)
    except psutil.NoSuchProcess:
        pass
    (Path(run.rundir) / "pid").unlink(missing_ok=True)
    return True


@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    """Enforce local mutation origins and guard runner paths before route dispatch.

    @ai-edited
    """
    host = request.headers.get("host", "")
    if not re.fullmatch(r"(?:localhost|127\.0\.0\.1|\[::1\])(?::[0-9]{1,5})?", host):
        return JSONResponse({"detail": "Local Host required"}, status_code=HTTPStatus.FORBIDDEN)
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        # Mutating requests from browsers must be same-origin, except the local Vite dev server.
        dev_origin = re.fullmatch(r"http://(?:localhost|127\.0\.0\.1):517[3-9]", origin or "")
        if origin and origin != f"http://{host}" and not dev_origin:
            return JSONResponse({"detail": "Cross-origin mutation forbidden"}, status_code=HTTPStatus.FORBIDDEN)
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "Cross-origin mutation forbidden"}, status_code=HTTPStatus.FORBIDDEN)
    if request.scope["path"].startswith("/runner/new/"):
        try:
            request.scope["path"] = "/runner/new/" + safe_log_path(request.scope["path"][len("/runner/new/") :])
        except HTTPException as exc:
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    try:
        return await call_next(request)
    except Exception as exc:
        logger.exception("Unhandled exception while handling %s %s", request.method, request.url)
        return error_response(exc)


app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(?:localhost|127\.0\.0\.1)(?::(?:5000|517[3-9]))?$",
    allow_credentials=False,
    allow_methods=["GET", "HEAD", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)


def register_routers():
    """Register API handlers before the static SPA fallback.

    @ai-edited
    """
    # Register API routers before declaring the SPA fallback route.
    from .experiments import router as experiment_router
    from .results import router as results_router
    from .runners import router as runner_router
    from .runs import router as runs_router
    from .system_info import router as system_info_router

    app.include_router(experiment_router)
    app.include_router(runs_router)
    app.include_router(results_router)
    app.include_router(runner_router)
    app.include_router(system_info_router)


register_routers()


def run(port: int, debug=False):
    """Serve the UI on loopback only.

    @ai-edited
    """
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=port, reload=debug, log_level="debug")


@app.get("/")
def index():
    """Serve the built UI entrypoint.

    @ai-edited
    """
    return FileResponse(dist_dir / "index.html")


@app.get("/{path:path}")
def spa_fallback(path: str):
    """Serve only files inside dist; unknown UI paths fall back to the entrypoint.

    @ai-edited
    """
    if path == "api" or path.startswith("api/"):
        raise HTTPException(HTTPStatus.NOT_FOUND, "API route not found")
    root = dist_dir.resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise HTTPException(HTTPStatus.FORBIDDEN, "Path is outside the UI distribution")
    if target.is_file():
        return FileResponse(target)
    return FileResponse(root / "index.html")
