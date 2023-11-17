from flask import Flask
from flask_cors import CORS
from ..server_state import ServerState

state = ServerState()


app = Flask(__name__, static_folder="/workspaces/marl/src/ui/dist/", static_url_path="")
CORS(app)


def run(port: int, debug=False):
    from . import runners
    from . import results
    from . import experiments

    try:
        app.run(port=port, debug=debug)
    except KeyboardInterrupt:
        print("Shutting down server...")
        exit(0)


@app.route("/")
def index():
    return app.send_static_file("index.html")
