from flask import Flask
from flask_cors import CORS

app = Flask(__name__, static_folder="/workspaces/marl/src/ui/dist/", static_url_path="")
CORS(app)


def run(port=5000, debug=False):
    from . import routes
    from . import system_info

    try:
        system_info.run(port + 1)
        app.run(port=port, debug=debug)
    except KeyboardInterrupt:
        print("Shutting down server...")
        exit(0)
