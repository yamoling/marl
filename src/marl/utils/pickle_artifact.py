import hashlib
import pickle
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

from .serialization import Serializable, serialization_root

ARTIFACT_DIRECTORY = "artifacts"


@dataclass
class PickleArtifact[T](Serializable):
    """Reference to an immutable, content-addressed pickle stored beside a JSON specification."""

    relative_path: str
    checksum: str
    count: int
    _value: T | None = field(init=False, default=None, repr=False, compare=False)
    _root: Path | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self):
        self._root = serialization_root()

    @classmethod
    def create(cls, value: T, *, count: int) -> Self:
        """Create an artifact that will be persisted with its owning specification. @ai-generated"""
        artifact = cls(relative_path="", checksum="", count=count)
        artifact._value = value
        return artifact

    @property
    def path(self) -> Path:
        """Return the resolved artifact path. @ai-generated"""
        if self._root is None or not self.relative_path:
            raise RuntimeError("The pickle artifact has not been associated with a saved specification.")
        return self._root / self.relative_path

    def materialize(self, root: Path, *, _seen: set[int] | None = None):
        """Persist a pending value or rebind an existing artifact to a specification root. @ai-generated"""
        root = root.resolve()
        if self.relative_path:
            destination = root / self.relative_path
            if destination.exists():
                self._root = root
                return
            if self._root is None:
                raise FileNotFoundError(destination)
            source = self._root / self.relative_path
            if not source.exists():
                raise FileNotFoundError(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            self._root = root
            return

        if self._value is None:
            raise RuntimeError("Cannot materialize a pickle artifact without a value.")

        artifact_dir = root / ARTIFACT_DIRECTORY
        artifact_dir.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=artifact_dir, prefix=".pickle-", delete=False) as temporary:
                temporary_path = Path(temporary.name)
                pickle.dump(self._value, temporary, protocol=pickle.HIGHEST_PROTOCOL)
            checksum = _sha256(temporary_path)
            relative_path = Path(ARTIFACT_DIRECTORY, f"{checksum}.pkl")
            destination = root / relative_path
            if destination.exists():
                temporary_path.unlink()
            else:
                temporary_path.replace(destination)
            self.relative_path = relative_path.as_posix()
            self.checksum = checksum
            self._root = root
            self._value = None
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def load(self) -> T:
        """Load and process-locally cache the artifact after verifying its digest. @ai-generated"""
        if self._value is not None:
            return self._value
        path = self.path
        actual_checksum = _sha256(path)
        if actual_checksum != self.checksum:
            raise ValueError(f"Checksum mismatch for pickle artifact {path}: expected {self.checksum}, got {actual_checksum}.")
        with path.open("rb") as artifact_file:
            self._value = pickle.load(artifact_file)
        return self._value


def _sha256(path: Path) -> str:
    """Compute a file digest without loading the full artifact into memory. @ai-generated"""
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        while chunk := artifact_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
