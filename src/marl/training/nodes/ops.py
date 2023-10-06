from .node import Node, T


class Add(Node[T]):
    """Addition operation"""

    def __init__(self, n1: Node[T], n2: Node[T]) -> None:
        super().__init__([n1, n2])
        self.n1 = n1
        self.n2 = n2

    def _compute_value(self) -> T:
        return self.n1.value + self.n2.value
