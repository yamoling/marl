from .node import Node, T


class Add(Node[T]):
    """Addition operation"""

    def __init__(self, *operands: Node[T]):
        super().__init__([*operands])

    def _compute_value(self) -> T:
        return sum(p.value for p in self.parents)  # type: ignore
