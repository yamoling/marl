from abc import ABC
from .exceptions import MissingParameterException


class Summarizable(ABC):
    """An abstract class for summarizable objects."""

    def summary(self) -> dict[str, ]:
        """Return a summary of the object."""
        return {
            "name": self.__class__.__name__
        }

    @classmethod
    def from_summary(cls, summary: dict[str, ]) -> "Summarizable":
        """Return an instance of the class from a summary."""
        try:
            return cls(**summary)
        except TypeError as e:
            raise MissingParameterException(e, cls.__name__)