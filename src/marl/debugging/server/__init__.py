from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

from .train import TrainServerState
from .replay import ReplayServerState

replay_state = ReplayServerState(None)
train_state = TrainServerState()


from .routes import upload_file

def run(port=5174, debug=False):
    app.run("0.0.0.0", port=port, debug=debug)
