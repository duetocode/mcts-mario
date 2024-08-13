from typing import Any, Callable, List, Tuple
import os
import gymnasium as gym
from monte_carlo_tree_search import Node, select, expand, rollout, backpropagate
import time
from multiprocessing import Pool
from tree_metrics import MeasureTree
import tqdm

_simulation_env = None


def _initialize_env(env_provider: Callable[[bool], gym.Env]):
    global _simulation_env

    _simulation_env = env_provider(
        render_mode="rgb_array", headless=True, with_reward=True
    )
    _simulation_env.reset()


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

    def __init__(self, env_provider: Callable[[], gym.Env], num_workers: int = None):
        self._env_provider = env_provider
        self.num_workers = num_workers if num_workers else os.cpu_count()
        self._pool = Pool(
            processes=self.num_workers,
            initializer=_initialize_env,
            initargs=[env_provider],
        )
        self._previous_node = None

    def act(self, env: gym.Env, observation: Any) -> Tuple[Any, Any]:
        """Select an action based on the given state"""
        # search for the optimal action
        action, tree = self._search(env)

        return action, tree

    def _search(self, env: gym.Env) -> Tuple[Any, Node]:
        """Perform a Monte Carlo Tree Search from the given state and select the best action."""
        t = time.time()

        # prepare the root node
        if self._previous_node is not None:
            # resue the previous node
            root_node = self._previous_node
        else:
            # create a new node
            root_node = Node(state=env.serialize(), action=0, value=0)

        # run the MCTS algorithm loop on the internal simulation environment
        depth = 0
        i = 0
        target_depth = 16
        target_num_nodes = 1200
        measure = MeasureTree()
        measure(root_node)
        num_nodes = measure.num_nodes
        pbar = tqdm.tqdm(total=target_depth)
        # while depth < target_depth or i < 8:
        while (num_nodes < target_num_nodes or depth < target_depth) and (
            num_nodes < target_num_nodes * 2 and depth < target_depth * 2
        ):
            i += 1
            # Selection
            candidates = select(root_node, exploration_weight=1.4)
            if len(candidates) == 0:
                break

            # Expansion
            new_nodes = expand(
                candidates,
                num_actions=env.action_space.n,
                max_expansions=self.num_workers,
            )
            num_nodes += len(new_nodes)

            if len(new_nodes) == 0:
                print("No more actions to explore")
                break

            # Rollout
            t_0 = time.time()
            # prepare new_nodes for rollout so that it will not transfer the whole tree
            nodes = []
            for node in new_nodes:
                parent = None
                if node.parent:
                    parent = Node(action=node.parent.action, state=node.parent.state)
                nodes.append(Node(action=node.action, parent=parent))

            rollout_results = self._pool.map(_rollout, nodes)
            depth = max([c[1] + 1 for c in candidates])
            pbar.set_description(
                f"Time: {time.time() - t_0:.4f} Depth: {depth} Nodes: {num_nodes}"
            )
            pbar.update(max(pbar.n, depth) - pbar.n)
            # Backpropagation
            for node, (state, is_terminated, rewards) in zip(
                new_nodes, rollout_results
            ):
                node.state = bytes(state)
                node.is_terminal = is_terminated
                backpropagate(node, list(rewards))
            del rollout_results

        pbar.close()
        # select the best action
        decision = max(root_node.children, key=lambda x: x.value)
        decision.parent = None

        print(
            f"Decision: {decision.action} {root_node.value} in {time.time() - t} seconds"
        )

        # display the tree
        measure = MeasureTree()
        measure(root_node)
        print(f"Number of nodes: {measure.num_nodes}")
        print(f"Max depth: {measure.max_depth}")

        # save the current node
        self._previous_node = decision

        return decision.action, root_node


class RolloutWorker:

    def __init__(self, env_provider: Callable[[], gym.Env]):
        self._simulation_env = env_provider()

    def rollout(self, node: Node) -> Tuple[Node, List[float]]:
        rewards = rollout(node, self._simulation_env)
        return node, rewards
