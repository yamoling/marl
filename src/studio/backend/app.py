import logging
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import security, settings
from .data.library import Library
from .errors import ApiError, error_response, not_found
from .services.events import EventHub, EventsConfig

logger = logging.getLogger(__name__)


def _install_error_handlers(app: FastAPI):
    """Every error body is `{error, message, issue?}`. @ai-generated"""

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError):
        return error_response(exc.error, exc.message, exc.status, exc.issue, exc.detail)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        errors = "; ".join(f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg', '')}" for e in exc.errors())
        return error_response("invalid-request", errors or "Invalid request", HTTPStatus.BAD_REQUEST)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        error = {404: "not-found", 403: "forbidden", 405: "method-not-allowed"}.get(exc.status_code, "http-error")
        return error_response(error, str(exc.detail), exc.status_code)


def create_app(
    root: Path | str | None = None,
    *,
    library: Library | None = None,
    events: EventsConfig | None = None,
    dist_dir: Path | None = None,
    system_interval: float = settings.SYSTEM_WS_INTERVAL_S,
) -> FastAPI:
    """
    Build the MARL Studio app: one `Library` (default: `settings.logs_root()`), one event hub,
    security middleware, API routers, then the SPA fallback.

    @ai-generated
    """
    library = library or Library(root if root is not None else settings.logs_root(), health_timeout=settings.HEALTH_TIMEOUT_S)
    hub = EventHub(library, events)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        hub._close_now()

    app = FastAPI(title="MARL Studio", lifespan=lifespan)
    app.state.library = library
    app.state.events = hub
    app.state.system_interval = system_interval
    app.state.dist_dir = dist_dir or settings.DIST_DIR
    security.install(app)
    _install_error_handlers(app)

    @app.get("/api/health")
    def health():
        """Liveness probe. @ai-generated"""
        return {"ok": True}

    register_routers(app)

    @app.get("/{path:path}")
    def spa(path: str):
        """
        Serve the built frontend; unknown non-API paths fall back to `index.html`.

        @ai-generated
        """
        if path == "api" or path.startswith("api/"):
            raise not_found("API route not found")
        root_dir: Path = app.state.dist_dir
        if not (root_dir / "index.html").is_file():
            return HTMLResponse(
                "<p>MARL Studio frontend is not built. Run <code>npm run build</code> in <code>src/studio/frontend</code>.</p>"
            )
        target = security.safe_dist_file(root_dir, path)
        return FileResponse(target if target is not None else root_dir.resolve() / "index.html")

    return app


def register_routers(app: FastAPI):
    """Register API routers before the SPA fallback. @ai-generated"""
    from .routes import events, experiments, runs, series, system

    for module in (experiments, series, runs, events, system):
        app.include_router(module.router)


app = create_app()
