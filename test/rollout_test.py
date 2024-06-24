import pytest
from unittest.mock import MagicMock
from monte_carlo_tree_search import rollout, Node


def test_rollout_with_non_terminal_node():
    """
    Test rollout with a non-terminal node.
    Should simulate until the end and return rewards.
    """
    # Setup the environment to return specific values
    mock_env = MagicMock()
    mock_env.step.side_effect = [
        ("state", 1, False, False, {}),
        ("state", 2, True, False, {}),
    ]
    # Prepare the node
    root_node = Node(state="root_state", parent=None)
    root_node.add(node := Node(action=1))
    assert node.parent == root_node

    # run the rollout
    rewards = rollout(root_node, mock_env)

    # the node should not be marked as visited because this will be done in the backpropagate function
    assert node.visits == 0, "Node should not be marked as visited."
    # check the rewards
    assert rewards == [1, 2], "Rewards should match the simulated values."


def test_rollout_with_terminal_node():
    """
    Test rollout with a terminal node.
    Should return an empty list of rewards.
    """
    mock_env = MagicMock()
    node = Node(is_terminal=True)

    rewards = rollout(node, mock_env)

    assert rewards == [], "Rewards list should be empty for a terminal node."
    assert mock_env.step.call_count == 0, "Environment should not be called."


def test_rollout_updates_node_state_and_termination():
    """
    Test that rollout updates the node's state and termination status.
    """
    mock_env = MagicMock()
    mock_env.step.return_value = ("state", 1, True, False, {})
    mock_env.save_state.side_effect = ["state-1", "state-2", "state-3"]
    # Prepare the node
    root_node = Node(state="root_state", parent=None)
    root_node.add(node := Node(action=1))

    rewards = rollout(node, mock_env)

    assert node.state == "state-1", "Node's state should be updated."
    assert node.is_terminal == True, "Node should be marked as terminal."


def test_rollout_with_random_actions_until_done():
    """
    Test rollout continues with random actions until the game is done.
    """
    mock_env = MagicMock()
    mock_env.step.side_effect = [
        ("state", 1, False, False, {}),
        ("state", 2, False, False, {}),
        ("state", 3, True, False, {}),
    ]
    mock_env.save_state.side_effect = ["state-1", "state-2", "state-3"]
    # Prepare the node
    root_node = Node(state="root_state", parent=None)
    root_node.add(node := Node(action=1))
    assert node.parent == root_node

    rewards = rollout(node, mock_env)

    assert rewards == [1, 2, 3], "Rewards should match the simulated values."
    assert node.is_terminal == False, "Node should not be marked as terminal."


def test_rollout_with_node_has_state():
    """
    Test rollout should use the node's state if available.
    """
    mock_env = MagicMock()
    mock_env.step.side_effect = [
        ("state-1", 1, False, False, {}),
        ("state-2", 2, True, False, {}),
    ]
    # Prepare the node
    root_node = Node(state="root_state", parent=None, value=3)

    rewards = rollout(root_node, mock_env)

    assert (
        mock_env.load_state.call_args[0][0] == "root_state"
    ), "Environment should load the root state."
    assert rewards == [1, 2], "Rewards should match the simulated values."
    assert root_node.state == "root_state", "Node's state should not be updated."
    assert root_node.value == 3, "Node's value should not be updated."
    assert (
        root_node.is_terminal == False
    ), "Node's is_terminal attribute should not be changed."
