from abc import ABC, abstractmethod

class Trainable(ABC):
    @abstractmethod
    def udpate(self, time_step: int):
        """Update the trainable object."""
    