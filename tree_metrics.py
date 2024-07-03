from monte_carlo_tree_search import Node


class MeasureTree:

    def __init__(self):
        self.num_nodes = 0
        self.max_depth = 0
        self.longest_path = None

    def __call__(self, node: Node):
        self._measure(node, 0)

    def _measure(self, node: Node, depth: int):
        """Travserse the tree with Depth First Search and measure the metrics"""

        if self.max_depth < depth:
            self.longest_path = node
            self.max_depth = depth

        self.num_nodes += 1

        # exit condition
        if node.children is None or len(node.children) == 0:
            return

        # recursive call
        for child in node.children:
            self._measure(child, depth + 1)
