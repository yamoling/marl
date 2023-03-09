import sys 
from marl.debugging.server import run


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None

enable_flask_debug_mode = not debugger_is_active()

# static_path = os.path.join(os.getcwd(), "src", "debug-ui", "dist")
run(port=5000, debug=enable_flask_debug_mode)
