from functools import cached_property
from abc import abstractmethod, ABC

class PropTest(ABC):
    @abstractmethod
    @cached_property
    def prop(self) -> int:
        """Yolo"""

class Child(PropTest):
    @cached_property
    def prop(self) -> int:
        return 1
    



c = Child()
print(c.__dict__.items())
print(dir(c))
print(getattr(c, "prop"))
c.prop += 5
print(c.prop)