import monte_carlo_tree_search as mcts


def test_select_on_leaf_node():
    """The method should return the node itself when called on a leaf node, as there are no children to traverse"""
    leaf = mcts.Node(action=0)
    assert leaf.is_leaf()
    assert not leaf.is_fully_expanded(4)

    actual = mcts.select(leaf, action_space=4)
    assert len(actual) == 1
    candidate = actual[0]
    assert candidate[0] == leaf
    assert candidate[1] == 0
    assert candidate[2] == float("inf")


def test_select_on_single_child():
    """The method should return the node and its child when the node has a single child, ensuring basic traversal works"""
    root = mcts.Node(value=20, visits=2, action=0)
    root.add(child := mcts.Node(action=1))

    actual = mcts.select(root)

    assert len(actual) == 2
    n_child, n_root = actual

    # the child should have higher UCB1 value, encouraging the algorithm to explore depth first
    assert n_child[0] == child
    assert n_root[0] == root

    # the child node should be at depth 1 and its ucb1 score is inf
    assert n_child[1] == 1
    assert n_child[2] == float("inf")

    # the root node should be still at depth 0 and its ucb1 score is not inf
    assert n_root[1] == 0
    assert n_root[2] == 10.0


def test_select_on_multiple_children_with_different_ucb1_values():
    """The select method should correctly select the child with the highest UCB1 value among multiple children.
    This can be done by manually setting the visits and value attributes of child nodes to ensure they have different UCB1 values
    """
    root = mcts.Node(visits=3, value=1, action=0)
    root.add(child_1 := mcts.Node(action=1, visits=1, value=1))
    root.add(child_2 := mcts.Node(action=2, visits=2, value=1))

    selected_node = mcts.select(root, max_candidates=2)

    # 2 nodes should be select
    assert len(selected_node) == 2
    # they are the child nodes
    assert selected_node[0][0] == child_1
    assert selected_node[1][0] == child_2

    # try again with 3 nodes
    selected_node = mcts.select(root, max_candidates=3)
    assert len(selected_node) == 3
    assert selected_node[-1][0] == root


def test_select_with_deep_tree():
    """Test the select method with a deeper tree (more than two levels)
    to ensure it correctly traverses the full tree and only selects the expandable nodes.
    """
    root = mcts.Node(visits=6, action=1)

    # first layer
    root.add(child_1 := mcts.Node(action=1, visits=3, value=1))
    root.add(child_2 := mcts.Node(action=2, visits=2, value=10))
    root.add(child_3 := mcts.Node(action=2, visits=2, value=10))
    root.add(child_4 := mcts.Node(action=2, visits=2, value=20))

    # second layer
    child_1.add(grandchild_1 := mcts.Node(action=1, visits=1, value=2))
    child_1.add(grandchild_2 := mcts.Node(action=2, visits=1, value=3))
    child_2.add(grandchild_3 := mcts.Node(action=1, visits=1, value=5))

    # run the selection
    selected_nodes = mcts.select(root, max_candidates=8)

    # 7 node should be selected and the root node is left out because it is fully expanded
    assert len(selected_nodes)
    assert root not in [node for node, _, _ in selected_nodes]


def test_select_on_node_with_unvisited_children():
    """Test that the select method can handle nodes with unvisited children (where visits is 0),
    which should have an infinite UCB1 value, effectively prioritizing their selection
    """
    # the root
    root = mcts.Node(visits=2, action=1)

    # the two children with a unvisited one
    root.add(mcts.Node(action=1, visits=1, value=10))
    root.add(the_one := mcts.Node(action=2, visits=0, value=0))

    # the second child should be selected as the first one
    selected_nodes = mcts.select(root)
    assert selected_nodes[0][0] == the_one
    assert selected_nodes[0][2] == float("inf")


def test_select_on_tree_with_equal_ucb1_values():
    """Test how the select method behaves when multiple children have equal UCB1 values, ensuring it can still select a node to traverse to."""
    # the root
    root = mcts.Node(visits=4)

    # the three children with equal UCB1 values
    root.add(child_0 := mcts.Node(action=0, visits=1, value=1))
    root.add(child_1 := mcts.Node(action=1, visits=1, value=1))
    root.add(child_2 := mcts.Node(action=2, visits=1, value=1))
    root.add(child_3 := mcts.Node(action=3, visits=1, value=1))

    # the second child should be at first because it has higher weight
    selected_nodes = mcts.select(root)
    assert selected_nodes[0][0] == child_1
