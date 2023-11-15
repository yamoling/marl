from ..server_state import ServerState
from .. import app

state = ServerState()

from .runners import *
from .results import *
from .experiments import *


@app.route("/")
def index():
    return app.send_static_file("index.html")
