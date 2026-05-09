import io
import logging
from dataclasses import dataclass

import optuna

from marl.models.replay_memory import ReplayMemory, TransitionMemory
from marl.training.qtarget_updater import HardUpdate, SoftUpdate, TargetParametersUpdater
from marl.utils import suggest
from marl.utils.tuning import _is_abstract

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.disable(logging.WARNING)

# --- Verify _is_abstract detection ---
assert _is_abstract(TargetParametersUpdater), "TargetParametersUpdater should be abstract"
assert not _is_abstract(SoftUpdate), "SoftUpdate should be concrete"
assert not _is_abstract(HardUpdate), "HardUpdate should be concrete"
assert _is_abstract(ReplayMemory), "ReplayMemory should be abstract"
assert not _is_abstract(TransitionMemory), "TransitionMemory should be concrete"
print("_is_abstract checks: OK")

# --- Test 1: suggest concrete SoftUpdate ---
study = optuna.create_study()
trial = study.ask()
updater = suggest(SoftUpdate, trial)
assert isinstance(updater, SoftUpdate)
assert 1e-3 <= updater.tau <= 0.5
print(f"SoftUpdate.tau = {updater.tau:.6f}  OK")

# --- Test 2: suggest concrete TransitionMemory ---
trial2 = study.ask()
mem = suggest(TransitionMemory, trial2)
assert isinstance(mem, TransitionMemory)
assert 1_000 <= mem.max_size <= 200_000
print(f"TransitionMemory.max_size = {mem.max_size}  OK")


# --- Test 3: rule 9 via a wrapping dataclass ---
@dataclass
class Wrapper:
    updater: TargetParametersUpdater


study3 = optuna.create_study()
trial3 = study3.ask()

# Intercept categorical choice to force HardUpdate
orig_cat = trial3.suggest_categorical


def force_hard(name, choices):
    if name.endswith(".__type__"):
        return "HardUpdate"
    return orig_cat(name, choices)


trial3.suggest_categorical = force_hard

wrapper = suggest(Wrapper, trial3)
assert isinstance(wrapper.updater, HardUpdate), f"Expected HardUpdate, got {type(wrapper.updater)}"
assert 50 <= wrapper.updater.update_period <= 2000
print(f"Rule 9 (abstract field): HardUpdate.update_period = {wrapper.updater.update_period}  OK")

# --- Test 4: rule 10 warning fires for undecorated float ---
logging.disable(logging.NOTSET)

handler = logging.StreamHandler(io.StringIO())
handler.setLevel(logging.WARNING)
logging.getLogger("marl.utils.tuning").addHandler(handler)


@dataclass
class WithGamma:
    gamma: float = 0.99


trial4 = study.ask()
result = suggest(WithGamma, trial4)
assert result.gamma == 0.99
warn_output = handler.stream.getvalue()
assert "gamma" in warn_output and "no tuning()" in warn_output
print(f"Rule 10 warning: {warn_output.strip()}  OK")

print()
print("All tests passed.")
