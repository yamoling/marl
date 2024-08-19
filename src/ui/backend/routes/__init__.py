import os
from flask import Flask
from flask_cors import CORS
from ..server_state import ServerState

state = ServerState()

cwd = os.getcwd()
dist_dir = ""
if os.path.basename(cwd) == "marl":
    dist_dir = os.path.join(cwd, "src/ui/dist/")
elif os.path.basename(cwd) == "src":
    dist_dir = os.path.join(os.getcwd(), "ui/dist/")
elif os.path.basename(cwd) == "ui":
    dist_dir = os.path.join(cwd, "dist/")
elif os.path.basename(cwd) == "backend":
    dist_dir = os.path.join(os.getcwd(), "../dist/")

if dist_dir == "" or not os.path.exists(dist_dir):
    raise RuntimeError("Could not find front end files to serve ! Make sure you have built them (cf: readme).")
dist_dir = os.path.join(os.getcwd(), "src/ui/dist/")

app = Flask(__name__, static_folder=dist_dir, static_url_path="")
CORS(app)


def run(port: int, debug=False):
    # Required to import these files without using them to register the flask routes.
    from . import runners
    from . import results
    from . import experiments
    from . import runs

    try:
        app.run(port=port, debug=debug)
    except KeyboardInterrupt:
        print("Shutting down server...")
        exit(0)


@app.route("/")
def index():
    return app.send_static_file("index.html")
