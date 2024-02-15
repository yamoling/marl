from abc import ABC, abstractmethod


class Updatable(ABC):
    @abstractmethod
    def update(self, time_step: int) -> dict[str, float]:
        """Update the object and return the corresponding logs."""
