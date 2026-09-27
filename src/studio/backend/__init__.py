import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run(port: int = 5000, root: Path | str | None = None):
    """Serve MARL Studio on loopback using the chosen experiment root. @ai-generated"""
    import uvicorn

    from .app import create_app

    try:
        uvicorn.run(create_app(root), host="127.0.0.1", port=port, log_level="info")
    except KeyboardInterrupt:
        logger.info("Shutting down MARL Studio...")
