import monte_carlo_tree_search as mcts


def test_expand_root_node():
    """Verify that when expand is called on the root node, it correctly adds all possible child nodes based on the number of actions."""
    node = mcts.Node(visits=0, is_terminal=False)

    children = mcts.expand([(node, 1, 12.0)], num_actions=4, max_expansions=10)

    # the expansion should only returns 2 nodes because of the limitation
    assert len(children) == 4
    assert [c.action for c in children] == [0, 1, 2, 3]
    # also check the children of the node
    assert children == node.children


def test_expand_root_node_with_limitation():
    """Verify that the max_expansions set to the expansion"""
    node = mcts.Node(visits=0, is_terminal=False)

    children = mcts.expand([(node, 1, 12.0)], num_actions=4, max_expansions=2)

    # the expansion should only returns 2 nodes because of the limitation
    assert len(children) == 2
    assert [c.action for c in children] == [0, 1]


def test_expand_multiple_nodes():
    """Verify that the expand function should limit its expansion when there are too many nodes"""
    root = mcts.Node(visits=10, is_terminal=False)
    root.add(child_1 := mcts.Node(visits=1, value=10))
    root.add(child_2 := mcts.Node(visits=1, value=5))

    new_nodes = mcts.expand(
        [(child_1, 3, 10), (child_2, 3, 8)],
        num_actions=4,
        max_expansions=6,
    )

    # there should be only 6 nodes in the `new_nodes` due to the max_expansions
    assert len(new_nodes) == 6
    # the child_1 should be fully expanded
    assert child_1.is_fully_expanded(action_space=4)
    # and the child_2 should only has two children
    assert len(child_2.children) == 2
    assert not child_2.is_fully_expanded(action_space=4)


def test_expand_terminal_node():
    """The function should ignore terminal nodes"""
    root = mcts.Node(visits=10, is_terminal=False)
    root.add(child_1 := mcts.Node(visits=1, value=10, is_terminal=True))
    root.add(child_2 := mcts.Node(visits=1, value=5))

    # expand
    new_nodes = mcts.expand(
        [(child_1, 3, 10), (child_2, 3, 8)],
        num_actions=4,
        max_expansions=6,
    )

    # there should only four nodes created becuase the child_1 should be ignored
    assert len(new_nodes) == 4
    # and these are all belong to child_2
    assert new_nodes == child_2.children
    # child_1 should be left untouched
    assert len(child_1.children) == 0
