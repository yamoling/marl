import sys 
import rlenv
from marl.utils.env_pool import EnvPool
from ui.backend import run


rlenv.register_wrapper(EnvPool)

def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None

run(port=5000, debug=False)
