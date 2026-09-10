import logging

import uvicorn

logger = logging.getLogger(__name__)


def run(port: int = 5000):
    from .routes import app

    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
