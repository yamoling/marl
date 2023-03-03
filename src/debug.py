import sys 
from marl.debugging.server import run


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None

enable_flask_debug_mode = not debugger_is_active()
run(port=5174, debug=enable_flask_debug_mode)
