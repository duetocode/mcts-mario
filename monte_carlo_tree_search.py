from typing import Any, List, Callable, Tuple
import math


class Node:
    """Monte Carlo Tree Search Node."""

    def __init__(
        self,
        action: Any = None,
        state: Any | None = None,
        parent: "Node" = None,
        children: List["Node"] = [],
        visits: int = 0,
        value: float = 0,
        is_terminal: bool = False,
    ):
        self.action = action
        self.state = state
        self.parent = parent
        self.children = children.copy()
        self.visits = visits
        self.value = value
        self.is_terminal = is_terminal

    def is_leaf(self) -> bool:
        """Check if the node is a leaf node."""
        return len(self.children) == 0

    def add(self, child: "Node"):
        """Add a child node to the current node."""
        child.parent = self
        self.children.append(child)


def calculate_ucb1(node: Node, exploration_weight: float = 1.41) -> float:
    """Calculate the Upper Confidence Bound 1 (UCB1) value for the given node."""

    if node.visits == 0:
        return float("inf")

    total_visits = sum(child.visits for child in node.parent.children)

    return node.value / node.visits + exploration_weight * math.sqrt(
        2 * math.log(total_visits) / node.visits
    )


def select(node: Node) -> Node:
    "Traversal a tree and select the most promising node."
    current_node = node
    while not current_node.is_leaf():
        # Select the child node with the highest UCB1 value
        current_node = max(current_node.children, key=calculate_ucb1)

    return current_node


def expand(node: Node, actions: List[Any]) -> List[Node]:
    """Expand the given node by adding all possible child nodes."""

    if node.is_terminal:
        return []

    new_nodes = []
    for action in actions:
        child = Node(action=action)
        node.add(child)
        new_nodes.append(child)

    return new_nodes


def rollout(
    node: Node, simulate_game: Callable[[Any], Tuple[Any, bool, List[int]]]
) -> List[int]:
    """Simulate a game from the given node until the end."""
    node.state, node.is_terminated, rewards = simulate_game(node.state)
    return rewards


def backpropagate(
    node: Node, rewards: List[int], reward_discount: float = 0.9
) -> float:
    """Update the value of the nodes in the path from the given node to the root."""
    # Firstly, backpropagate the rewards from the terminal node to the node
    cumulative_reward = 0.0
    for reward in reversed(rewards):
        cumulative_reward = reward + reward_discount * cumulative_reward

    # Then, update the value of the nodes in the path from the given node to the root
    current_node = node
    while current_node is not None:
        current_node.visits += 1
        current_node.value += cumulative_reward
        current_node = current_node.parent

    return cumulative_reward
