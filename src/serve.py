import sys 
from marl.server import run


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None

debug = debugger_is_active()
dev = False

# static_path = os.path.join(os.getcwd(), "src", "debug-ui", "dist")
run(port=5000, debug=debug or dev)
