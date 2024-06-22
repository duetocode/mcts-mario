import monte_carlo_tree_search as mcts


def test_single_node():
    """Verify that the function correctly updates the value and visits of a single node without a parent"""
    node = mcts.Node()

    mcts.backpropagate(node, [10])

    assert node.visits == 1
    assert node.value == 10


def test_linear_tree():
    """Create a simple linear tree (a node with a parent, and that parent might have its own parent, etc.)
    and verify that the function updates the value and visits of each node in the path correctly.
    """
    root = mcts.Node(visits=2, value=2)
    root.add(node_1 := mcts.Node(visits=1, value=1))
    node_1.add(node_2 := mcts.Node())

    rewards = [5, 10, 25]

    mcts.backpropagate(node_2, rewards)

    assert node_2.value == 34.25
    assert node_2.visits == 1
    assert node_1.value == 35.25
    assert node_1.visits == 2
    assert root.value == 36.25
    assert root.visits == 3
