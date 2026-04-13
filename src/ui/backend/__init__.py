import logging
import uvicorn


def run(port: int = 5000):
    from .routes import app

    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    except KeyboardInterrupt:
        logging.info("Shutting down server...")
        exit(0)
