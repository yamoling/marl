from abc import ABC, abstractmethod
from typing import List, Generic, TypeVar
import torch

T = TypeVar("T")


class Node(Generic[T], ABC):
    """
    Parent class of any node.

    A Node has a value, which is computed from its parents' values, except if it is a `ValueNode`.
    Whenever a node is marked for update, it marks all its children for update as well but does not perform the actual udpate.
    The update is only performed when the value of a node is requested and it is marked for update.
    """

    num = 0

    def __init__(self, parents: List["Node"]):
        if len(parents) == 0:
            self.level = 0
        else:
            self.level = max([p.level for p in parents]) + 1
        self.children: List[Node] = []
        self.parents: List[Node] = parents
        for parent in parents:
            parent.children.append(self)
        self.name = f"{self.__class__.__name__}-{Node.num}"
        Node.num += 1

        self._needs_update = True
        # Type hinting. self._cache is initially unbound.
        self._cache: T

    @abstractmethod
    def _compute_value(self) -> T:
        """Compute the node value."""

    def _mark_for_update(self):
        # If the current node is already marked for update, so are its children
        if self._needs_update:
            return
        for child in self.children:
            child._mark_for_update()
        self._needs_update = True

    def to(self, device: torch.device):
        """Move the node to the given device"""
        for child in self.children:
            child.to(device)

    def randomize(self):
        """Randomize the node value"""
        for child in self.children:
            child.randomize()

    @property
    def value(self) -> T:
        """The value of the node"""
        if self._needs_update:
            self._cache = self._compute_value()
            self._needs_update = False
        return self._cache

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

        # Operator overloading

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return self.value * other

    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return self.value + other

    def __sub__(self, other):
        return self.value - other

    def __rsub__(self, other):
        return other - self.value

    def __div__(self, other):
        return self.value / other

    def __rdiv__(self, other):
        return other / self.value

    def __truediv__(self, other):
        return self.value / other

    def __rtruediv__(self, other):
        return other / self.value

    def __lt__(self, other) -> bool:
        return self.value < other

    def __le__(self, other) -> bool:
        return self.value <= other

    def __gt__(self, other) -> bool:
        return self.value > other

    def __ge__(self, other) -> bool:
        return self.value >= other

    def __eq__(self, other) -> bool:
        return self.value == other

    def __ne__(self, other) -> bool:
        return self.value != other


class ValueNode(Node[T]):
    """Constant value node"""

    def __init__(self, value: T):
        super().__init__([])
        self._cache = value
        self._needs_update = False

    def _compute_value(self):
        raise RuntimeError("ValueNode should not be updated")

    @property
    def value(self):
        return self._cache

    @value.setter
    def value(self, new_value: T):
        self._cache = new_value
        for child in self.children:
            child._mark_for_update()
