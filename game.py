from typing import Any, List
import gymnasium as gym
from monte_carlo_tree_search import Node, select, expand, rollout, backpropagate


class AgentKane:

    def __init__(self, env: gym.Env):
        self._simulation_env = gym.make("super-mario-bros-1-1-v0")

    def act(self, env: gym.Env):
        """Select an action based on the given state"""
        # search for the optimal action
        action = self._search(env)

        return action

    def _search(self, env: gym.Env) -> Any:
        """Perform a Monte Carlo Tree Search from the given state and select the best action."""
        # prepare the root node
        root_node = Node(state=env.save_state(), value=0)
        # run the MCTS algorithm loop on the internal simulation environment
        for i in range(1000):
            # Selection
            node = select(root_node)
            # Expansion
            node = expand(node)
            # Simulation
            rewards = rollout(node, self._simulation_env)
            # Backpropagation
            backpropagate(node, rewards)

        # select the best action
        return max(root_node.children, key=lambda x: x.value / max(1, x.visits)).action
