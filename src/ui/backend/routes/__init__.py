import os
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from ..server_state import ServerState

state = ServerState()

cwd = os.getcwd()
dist_dir = ""
if os.path.basename(cwd) == "marl":
    dist_dir = os.path.join(cwd, "src/ui/dist/")
elif os.path.basename(cwd) == "src":
    dist_dir = os.path.join(os.getcwd(), "ui/dist/")
elif os.path.basename(cwd) == "ui":
    dist_dir = os.path.join(cwd, "dist/")
elif os.path.basename(cwd) == "backend":
    dist_dir = os.path.join(os.getcwd(), "../dist/")

if dist_dir == "" or not os.path.exists(dist_dir):
    raise RuntimeError("Could not find front end files to serve ! Make sure you have built them (cf: readme).")
dist_dir = os.path.join(os.getcwd(), "src/ui/dist/")


app = FastAPI()
logger = logging.getLogger(__name__)


@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.exception("Unhandled exception while handling %s %s", request.method, request.url)
        return JSONResponse(
            status_code=500,
            content={"message": "Internal Server Error", "detail": str(exc)},
        )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def register_routers():
    # Register API routers before declaring the SPA fallback route.
    from .experiments import router as experiment_router
    from .results import router as results_router
    from .runners import router as runner_router
    from .runs import router as runs_router

    app.include_router(experiment_router)
    app.include_router(runs_router)
    app.include_router(results_router)
    app.include_router(runner_router)


register_routers()


def run(port: int, debug=False):
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port, reload=debug, log_level="debug")


@app.get("/")
def index():
    return FileResponse(os.path.join(dist_dir, "index.html"))


@app.get("/{path:path}")
def spa_fallback(path: str):
    target = os.path.join(dist_dir, path)
    if os.path.isfile(target):
        return FileResponse(target)
    return FileResponse(os.path.join(dist_dir, "index.html"))
