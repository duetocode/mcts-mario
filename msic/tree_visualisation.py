import graphviz
import cv2
from monte_carlo_tree_search import Node


def visualise(node: Node):
    """Visualise the tree and its statistical infomation."""
    dot = graphviz.Digraph()

    # Recursively add nodes to the graph
    counter = 0

    def add_node(node: Node) -> int:
        nonlocal counter
        node_id = counter
        counter += 1
        dot.node(str(id), label=f"{node.value:.2f}/{node.visits}")
        for child in node.children:
            child_id = add_node(child)
            dot.edge(str(node_id), str(child_id))
        return node_id

    add_node(node)

    # render the tree to image
    dot.render("tree", format="png", cleanup=True)

    tree_img = cv2.imread("tree.png")

    cv2.imshow("Tree", tree_img)
