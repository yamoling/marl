class CorruptExperimentException(Exception):
    """Raised when the experiment is corrupted and cannot be loaded."""

    pass


class ExperimentVersionMismatch(Exception):
    """Raised when the experiment is corrupted and cannot be loaded."""

    def __init__(self, faulty_key: str) -> None:
        super().__init__(f"You experiment version is outdated: key '{faulty_key}' is not found in summary !")


class EmptyForcedActionsException(Exception):
    """Raised when the experiment is corrupted and cannot be loaded."""

    def __init__(self) -> None:
        super().__init__("The experiment has no forced actions !")


class ExperimentAlreadyExistsException(Exception):
    def __init__(self, logdir: str):
        self.logdir = logdir
        super().__init__(f"The experiment {logdir} already exists: impossible to create a new one in the same directory!")


class TestEnvNotSavedException(Exception):
    pass


class AlreadyRunningException(Exception):
    def __init__(self, rundir: str, pid: int):
        super().__init__()
        self.pid = pid
        self.rundir = rundir

    def __repr__(self) -> str:
        return f"Run {self.rundir} is already running (pid {self.pid})"


class NotRunningExcception(Exception):
    def __init__(self, rundir: str):
        super().__init__()
        self.rundir = rundir

    def __repr__(self) -> str:
        return f"Run {self.rundir} is not running"


class RunProcessNotFound(Exception):
    def __init__(self, rundir: str, pid: int):
        super().__init__()
        self.pid = pid
        self.rundir = rundir

    def __repr__(self) -> str:
        return f"Run {self.rundir} with pid {self.pid} not found"
