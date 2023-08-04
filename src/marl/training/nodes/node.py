from abc import ABC, abstractmethod
from typing import List, Generic, TypeVar
from serde import serde, field
from dataclasses import dataclass

T = TypeVar("T")


# def node(cls):
#     """Decorator to make a Node class. It handles parent/child relationships."""
#     class NewClass(cls, Node[T]):
#         def __init__(self, *args, **kwargs):
#             cls.__init__(self, *args, **kwargs)
#             Node.__init__(self)
#             self.parents = []
#             self.name = f"{cls.__name__}-{Node.num}"
#             all_args = args + tuple(kwargs.values())
#             for arg in all_args:
#                 if isinstance(arg, Node):
#                     arg.children.append(self)
#                     self.parents.append(arg)
#     NewClass.__init__.__signature__ = inspect.signature(cls.__init__)
#     return NewClass


class Node(Generic[T], ABC):
    """Parent class of any node"""

    num = 0

    def __init__(self, parents: List["Node"]):
        self.children: List[Node] = []
        self.parents: List[Node] = parents
        for parent in parents:
            parent.children.append(self)
        self.name = f"{self.__class__.__name__}-{Node.num}"
        Node.num += 1

    @property
    @abstractmethod
    def value(self) -> T:
        """The value of the node"""

    def replace_parent(self, old_parent: "Node[T]", new_parent: "Node[T]"):
        """
        Replace a parent with an other node of the same type.
        """
        for key, value in self.__dict__.items():
            if value is old_parent:
                setattr(self, key, new_parent)
                # Bookkeeping:
                # - Remove the old parent from the parents list
                # - Remove self from the children list of the old parent
                # - Add the new parent to the parents list
                # - Add self to the children list of the new parent
                self.parents.remove(old_parent)
                self.parents.append(new_parent)
                old_parent.children.remove(self)
                new_parent.children.append(self)
                return
        raise ValueError("The given node is not a parent of self !")

    def replace(self, new_node: "Node[T]"):
        """Replace the current node by another one of the same type"""
        for child in self.children:
            child.replace_parent(self, new_node)

    def __hash__(self) -> int:
        return hash(self.name)


@serde
@dataclass(eq=False)
class ValueNode(Node[T]):
    """Constant value node"""

    _value: T = field(rename="value", serializer=lambda x: None, deserializer=lambda x: None)

    def __init__(self, value: T):
        super().__init__([])
        self._value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value: T):
        self._value = new_value
