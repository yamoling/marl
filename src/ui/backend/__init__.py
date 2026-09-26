import logging

import uvicorn

logger = logging.getLogger(__name__)


def run(port: int = 5000):
    """Serve the local UI on the loopback interface only.

    @ai-edited
    """
    from .routes import app

    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
