from .node import Node, T


class AddNode(Node[T]):
    """Addition operation"""

    def __init__(self, n1: Node[T], n2: Node[T]) -> None:
        super().__init__()
        self.n1 = n1
        self.n2 = n2
        n1.add_child(self)
        n2.add_child(self)

    @property
    def value(self) -> T:
        return self.n1.value + self.n2.value


class SubNode(Node[T]):
    """Subtration operation"""

    def __init__(self, n1: Node[T], n2: Node[T]) -> None:
        super().__init__()
        self.n1 = n1
        self.n2 = n2
        n1.add_child(self)
        n2.add_child(self)

    @property
    def value(self) -> T:
        return self.n1.value - self.n2.value