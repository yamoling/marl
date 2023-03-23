

class CorruptExperimentException(Exception):
    """Raised when the experiment is corrupted and cannot be loaded."""
    pass

class EmptyForcedActionsException(Exception):
    """Raised when the experiment is corrupted and cannot be loaded."""
    def __init__(self) -> None:
        super().__init__("The experiment has no forced actions !")