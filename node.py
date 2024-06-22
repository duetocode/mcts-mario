from typing import Any


class Node:
    """Monte Carlo Tree Search Node."""

    def __init__(
        self,
        action: Any = None,
        state: Any = None,
        parent: "Node" | None = None,
    ):
        self.action = action
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_leaf(self) -> bool:
        """Check if the node is a leaf node."""
        return len(self.children) == 0

    def add(self, child: "Node"):
        """Add a child node."""
        child.parent = self
        self.children.append(child)
