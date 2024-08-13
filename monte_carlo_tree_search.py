from typing import Any, List, Callable, Tuple
import math
from collections import deque

import gymnasium as gym

action_weights = [
    # ["NOOP"],
    0.5,
    # ["right", "B"],
    1.5,
    # ["right", "A", "B"],
    1.0,
    # ["left"]
    0.1,
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

    def is_fully_expanded(self, action_space: int):
        """Return True if the node has all possible children."""
        return len(self.children) == action_space


def ucb1(node: Node, exploration_weight: float) -> float:
    """Calculate the Upper Confidence Bound 1 (UCB1) value for the given node."""

    # prioritize the unvisited nodes
    if node.visits == 0:
        return float("inf")

    # for root node, the total visits is the same as the node's visits
    if node.parent is None:
        return node.value / node.visits

    # calculate the UCB1 value for regular nodes
    ucb1_value = node.value / node.visits + math.sqrt(
        2 * math.log(node.parent.visits) / node.visits
    )

    return ucb1_value * action_weights[node.action]


def select(
    node: Node,
    max_candidates: int = 8,
    exploration_weight: float = 1.0,
    action_space: int = 4,
) -> List[Tuple[Node, int, int]]:
    """Traversal a search tree and select the most promising node.
    parameters
    ----------
    node: Node
        The root node of the tree.
    max_candidates: int
        The max number of selected candidate nodes
    exploration_weight: float
        The exploration weight for the UCB1 calculation.

    return
    ------
    List[Tuple[Node, int, float]]
        The list of selected nodes with the depth of the node and their UCB1 score.
    """

    # 1. collect all nodes that are not fully expanded yet
    stack = [(node, 0)]
    candidates = []
    while stack:
        current_node, depth = stack.pop()
        stack.extend(map(lambda n: (n, depth + 1), current_node.children))
        # add nodes that can expand to the candidates for further short listing
        if not current_node.is_fully_expanded(action_space):
            candidates.append((current_node, depth))

    # 2. calculate the ucb1 scores for these nodes
    candidates = [
        (
            node,
            depth,
            ucb1(node, exploration_weight),
        )
        for node, depth in candidates
    ]
    # sort condidation: depth DESC, ucb1 DESC, then the action weight
    candidates.sort(
        # key=lambda x: (x[1], x[2], action_weights[x[0].action] if x[0].action else 0),
        key=lambda x: (x[2], action_weights[x[0].action]),
        reverse=True,
    )

    # 3. select the best candidates
    return candidates[:max_candidates]


def expand(
    candidates: List[Tuple[Node, int, float]], num_actions: int, max_expansions: int = 8
) -> List[Node]:
    """Expand the given nodes by adding all possible child nodes."""
    new_nodes = []

    # continue expand the nodes until the limit is reached or there is no expandable nodes left
    for node, _, _ in candidates:
        # ignore the terminal node
        if node.is_terminal:
            continue
        # expand the node
        for i in range(num_actions):
            # check with the limitation
            if len(new_nodes) >= max_expansions:
                return new_nodes
            # expand
            child = Node(action=i)
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
    for i, reward in enumerate(rewards):
        cumulative_reward += (reward_discount**i) * reward

    # Then, update the value of the nodes in the path from the given node to the root
    current_node = node
    i = 0
    while current_node is not None:
        current_node.visits += 1
        # apply the discounted reward to the node
        current_node.value += cumulative_reward * (reward_discount**i)
        current_node = current_node.parent
        i += 1

    return cumulative_reward
