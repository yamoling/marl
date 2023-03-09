from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

from .train import TrainServerState
from .replay import ReplayServerState

replay_state = ReplayServerState(None)
train_state = TrainServerState()

def run(port=5000, static_path: str=None, debug=False):
    from . import routes
    app.static_folder = static_path
    app.run(port=port, debug=debug)
