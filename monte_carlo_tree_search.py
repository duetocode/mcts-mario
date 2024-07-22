from typing import Any, List, Callable, Tuple
import math

import gymnasium as gym

action_weights = [
    # ["NOOP"],
    0.5,
    # ["right", "B"],
    1.5,
    # ["right", "A", "B"],
    1.0,
    # ["left"]
    0.5,
]


class Node:
    """Monte Carlo Tree Search Node."""

    def __init__(
        self,
        action: int = None,
        state: bytes | None = None,
        parent: "Node" = None,
        children: List["Node"] = [],
        visits: int = 0,
        value: float = 0,
        is_terminal: bool = False,
        is_victory: bool = False,
    ):
        self.action = action
        self.state = state
        self.parent = parent
        self.children = children.copy()
        self.visits = visits
        self.value = value
        self.is_terminal = is_terminal
        self.is_victory = is_victory

    def is_leaf(self) -> bool:
        """Check if the node is a leaf node."""
        return len(self.children) == 0

    def add(self, child: "Node"):
        """Add a child node to the current node."""
        child.parent = self
        self.children.append(child)


def calculate_ucb1(node: Node, exploration_weight: float) -> float:
    """Calculate the Upper Confidence Bound 1 (UCB1) value for the given node."""

    if node.visits == 0:
        return float("inf")

    total_visits = sum(child.visits for child in node.parent.children)

    ucb1_value = node.value / node.visits + exploration_weight * math.sqrt(
        2 * math.log(total_visits) / node.visits
    )

    return ucb1_value * action_weights[node.action]


def select(node: Node, exploration_weight: float = 1.0) -> Tuple[Node, int]:
    "Traversal a tree and select the most promising node."
    current_node = node
    depth = 1
    while not current_node.is_leaf():
        # Select the child node with the highest UCB1 value
        current_node = max(
            current_node.children, key=lambda n: calculate_ucb1(n, exploration_weight)
        )
        depth += 1

    return current_node, depth


def expand(node: Node, num_actions: int) -> List[Node]:
    """Expand the given node by adding all possible child nodes."""

    if node.is_terminal:
        return []

    new_nodes = []
    for action in range(num_actions):
        child = Node(action=action)
        node.add(child)
        new_nodes.append(child)

    return new_nodes


def rollout(node: Node, env: gym.Env) -> List[float]:
    """Simulate a game from the given node until the end. If the node is not simulated, simulate the game from the node."""
    # If the node is a terminal node, return an empty list
    if node.is_terminal:
        return []

    rewards = []

    env.reset()

    # load the state from the node if it is not None
    if node.state is not None:
        env.deserialize(node.state)
    # load the parent's state as the initial state
    # and run the node since it is not run yet
    elif node.parent:
        env.deserialize(node.parent.state)
        # run the node
        _, reward, terminated, truncated, info = env.step(node.action)
        rewards.append(reward)
        if info["flag_get"] == True:
            node.is_victory = True
        node.is_terminal = terminated or truncated
        node.state = env.serialize()
        if node.is_terminal:
            return rewards
    else:
        raise ValueError("The node is not a terminal node, but it does not have")

    # run the rest of the game with random actions
    done = False
    while not done:
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        rewards.append(reward)

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
        current_node.value += cumulative_reward * reward_discount
        current_node = current_node.parent

    return cumulative_reward
