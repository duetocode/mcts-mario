from typing import Any, Callable, List, Tuple
import os
import gymnasium as gym
from monte_carlo_tree_search import Node, select, expand, rollout, backpropagate
import time
from multiprocessing import Pool

_simulation_env = None


def _initialize_env(env_provider: Callable[[], gym.Env]):
    global _simulation_env
    _simulation_env = env_provider()


def _rollout(node: Node) -> Tuple[bytes, bool, List[float]]:
    """Run a single rollout from a given node and return the rewards and the terminate status.

    args:
        node: The node to start the rollout from.
    returns:
        bool: True if the node is terminal, False otherwise.
        List[float]: The rewards collected during the rollout.
    """
    # the node
    rewards = rollout(node, _simulation_env)

    # we return the is_terminal value because this function might run in sub process
    # where the node object is an copy from the original in the main process
    return node.state, node.is_terminal, rewards


class AgentKane:

    def __init__(self, env_provider: Callable[[], gym.Env]):
        self._env_provider = env_provider
        self._previous_decision = None
        self._pool = Pool(initializer=_initialize_env, initargs=[env_provider])

    def act(self, env: gym.Env, observation: Any):
        """Select an action based on the given state"""
        # search for the optimal action
        action = self._search(env)

        return action

    def _search(self, env: gym.Env) -> Any:
        """Perform a Monte Carlo Tree Search from the given state and select the best action."""
        t = time.time()
        # prepare the root node
        root_node = Node(state=env.serialize(), value=0)
        # run the MCTS algorithm loop on the internal simulation environment
        for i in range(21):
            # Selection
            node = select(root_node)
            # Expansion
            new_nodes = expand(node, env.action_space.n)
            # Rollout
            t_r = time.time()
            rollout_results = self._pool.map(_rollout, new_nodes)
            print(f"[{i}]Rollout for {len(new_nodes)}. in {time.time() - t_r: .3f} ")
            # Backpropagation
            for node, (state, is_terminated, rewards) in zip(
                new_nodes, rollout_results
            ):
                node.state = state
                node.is_terminal = is_terminated
                backpropagate(node, rewards)

        # select the best action
        decision = max(root_node.children, key=lambda x: x.value / max(1, x.visits))
        print(f"Decision: {decision.action} in {time.time() - t} seconds")

        # choose the best path for the next iteration

        return decision.action


class RolloutWorker:

    def __init__(self, env_provider: Callable[[], gym.Env]):
        self._simulation_env = env_provider()

    def rollout(self, node: Node) -> Tuple[Node, List[float]]:
        rewards = rollout(node, self._simulation_env)
        return node, rewards
