from flask import Flask
from flask_cors import CORS

app = Flask(__name__, static_folder="../dist/", static_url_path='')
CORS(app)

from .server_state import ServerState

state = ServerState()

def run(port=5000, debug=False):
    from . import routes
    try:
        import os
        print(os.getpid())
        app.run(port=port, debug=debug)
    except KeyboardInterrupt:
        print("Shutting down server...")
        exit(0)
