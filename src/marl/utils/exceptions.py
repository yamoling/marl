

class CorruptExperimentException(Exception):
    """Raised when the experiment is corrupted and cannot be loaded."""
    pass

class EmptyForcedActionsException(Exception):
    """Raised when the experiment is corrupted and cannot be loaded."""
    def __init__(self) -> None:
        super().__init__("The experiment has no forced actions !")

class ExperimentAlreadyExistsException(Exception):
    def __init__(self, logdir: str):
        super().__init__(f"The experiment {logdir} already exists: impossible to create a new one in the same directory!")
        