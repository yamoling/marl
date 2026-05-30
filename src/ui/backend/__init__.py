import logging

import uvicorn


def run(port: int = 5000):
    from .routes import app

    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except KeyboardInterrupt:
        logging.info("Shutting down server...")
