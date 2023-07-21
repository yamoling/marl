from abc import ABC, abstractmethod
from typing import Set, Generic, TypeVar

T = TypeVar("T")


class Node(Generic[T], ABC):
    """Parent class of any node"""

    num = 0

    def __init__(self) -> None:
        super().__init__()
        self.children: Set["Node"] = set()
        self.name = f"{self.__class__.__name__}-{Node.num}"
        Node.num += 1

    @property
    @abstractmethod
    def value(self) -> T:
        """The value of the node"""

    def insert_after(self, new_node: "Node[T]"):
        """Insert the new node after the current one"""
        new_node.children = self.children
        self.children = [new_node]
        for child in new_node.children:
            child.replace_parent(self, new_node)

    def replace_parent(self, old_parent: "Node[T]", new_parent: "Node[T]"):
        """
        Replace a parent with an other node of the same type.
        """
        for key, value in self.__dict__.items():
            if value is old_parent:
                setattr(self, key, new_parent)
                return
        raise ValueError("The given node is not a parent of self !")

    def replace_by(self, new_node: "Node[T]"):
        """Replace the current node by another one of the same type"""
        for child in self.children:
            child.replace_parent(self, new_node)
        new_node.children = self.children

    def add_child(self, *nodes: "Node"):
        """Add the given nodes as children"""
        for node in nodes:
            self.children.add(node)

    def __hash__(self) -> int:
        return hash(self.name)


class ValueNode(Node[T]):
    """Constant value node"""

    def __init__(self, value: T) -> None:
        super().__init__()
        self._value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value: T):
        self._value = new_value
