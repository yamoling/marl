import os
from flask import Flask
from flask_cors import CORS
from ..server_state import ServerState

state = ServerState()

cwd = os.getcwd()
if os.path.basename(cwd) == "marl":
    dist_dir = os.path.join(cwd, "src/ui/dist/")
elif os.path.basename(cwd) == "src":
    dist_dir = os.path.join(os.getcwd(), "ui/dist/")
elif os.path.basename(cwd) == "ui":
    dist_dir = os.path.join(cwd, "dist/")
elif os.path.basename(cwd) == "backend":
    dist_dir = os.path.join(os.getcwd(), "../dist/")
else:
    raise RuntimeError("Please run this script from the root of the project, otherwise")
dist_dir = os.path.join(os.getcwd(), "src/ui/dist/")

app = Flask(__name__, static_folder=dist_dir, static_url_path="")
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
