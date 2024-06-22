import monte_carlo_tree_search as mcts


def test_select_on_leaf_node():
    """The method should return the node itself when called on a leaf node, as there are no children to traverse"""
    leaf = mcts.Node()
    assert leaf.is_leaf()
    assert mcts.select(leaf) == leaf


def test_select_on_single_child():
    """The method should return the only child when the node has a single child, ensuring basic traversal works"""
    root = mcts.Node()
    root.add(child := mcts.Node(action=1))

    assert mcts.select(root) == child


def test_select_on_multiple_children_with_different_ucb1_values():
    """The select method should correctly select the child with the highest UCB1 value among multiple children.
    This can be done by manually setting the visits and value attributes of child nodes to ensure they have different UCB1 values
    """
    root = mcts.Node(visits=3, value=1)
    root.add(child_1 := mcts.Node(action=1, visits=1, value=1))
    root.add(child_2 := mcts.Node(action=2, visits=2, value=1))

    # according to the UCB1, child_1 should be selected
    selected_node = mcts.select(root)
    assert selected_node == child_1


def test_select_with_deep_tree():
    """Test the select method with a deeper tree (more than two levels) to ensure it correctly traverses down to a leaf node,
    selecting the highest UCB1 value at each level."""
    root = mcts.Node(visits=6)

    # first layer
    root.add(child_1 := mcts.Node(action=1, visits=3, value=1))
    root.add(child_2 := mcts.Node(action=2, visits=2, value=10))

    # the child_2 should be selected
    selected_node = mcts.select(root)
    assert selected_node == child_2

    # second layer
    child_1.add(grandchild_1 := mcts.Node(action=1, visits=1, value=2))
    child_1.add(grandchild_2 := mcts.Node(action=2, visits=1, value=3))
    child_2.add(grandchild_3 := mcts.Node(action=1, visits=1, value=5))

    # the grandchild_3 should be selected
    selected_node = mcts.select(root)
    assert selected_node == grandchild_3


def test_select_on_node_with_unvisited_children():
    """Test that the select method can handle nodes with unvisited children (where visits is 0),
    which should have an infinite UCB1 value, effectively prioritizing their selection
    """
    # the root
    root = mcts.Node(visits=2)

    # the two children with a unvisited one
    root.add(mcts.Node(action=1, visits=1, value=10))
    root.add(the_one := mcts.Node(action=2, visits=0, value=0))

    # the second child should be selected
    selected_node = mcts.select(root)
    assert selected_node == the_one


def test_select_on_tree_with_equal_ucb1_values():
    """Test how the select method behaves when multiple children have equal UCB1 values, ensuring it can still select a node to traverse to."""
    # the root
    root = mcts.Node(visits=4)

    # the three children with equal UCB1 values
    root.add(child_1 := mcts.Node(action=1, visits=1, value=1))
    root.add(child_2 := mcts.Node(action=2, visits=1, value=1))
    root.add(child_3 := mcts.Node(action=3, visits=1, value=1))

    # the first child should be selected
    selected_node = mcts.select(root)
    assert selected_node == child_1
