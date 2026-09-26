import logging

logger = logging.getLogger(__name__)


def run(port: int = 5000):
    """Serve MARL Studio on the loopback interface only. @ai-generated"""
    import uvicorn

    from .app import app

    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
    except KeyboardInterrupt:
        logger.info("Shutting down MARL Studio...")
