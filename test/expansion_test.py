import monte_carlo_tree_search as mcts


def test_expand_non_terminal_node():
    """Verify that when expand is called on a non-terminal node with a list of actions, it correctly adds all possible child nodes based on those actions."""
    node = mcts.Node(visits=0, is_terminal=False)
    actions = [1, 2, 3]

    children = mcts.expand(node, actions)

    assert len(children) == 3
    assert [c.action for c in children] == actions
    assert node.children == children


def test_expand_terminal_node():
    """Ensure that when expand is called on a terminal node, it returns an empty list, indicating no expansion."""
    node = mcts.Node(is_terminal=True)
    children = mcts.expand(node, [1, 2, 3])
    assert node.is_leaf()
    assert len(children) == 0
