from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

from .server_state import ServerState

state = ServerState()

def run(port=5000, static_path: str=None, debug=False):
    from . import routes
    app.static_folder = static_path
    try:
        import os
        print(os.getpid())
        app.run(port=port, debug=debug)
    except KeyboardInterrupt:
        print("Shutting down server...")
        exit(0)
