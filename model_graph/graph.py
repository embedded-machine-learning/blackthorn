"""The graph module"""

from collections import OrderedDict

import pickle
from graphviz import Digraph


class EstimationGraph:
    """Class to hold the graph for an estimation function"""

    def __init__(self):

        self._nodes = OrderedDict()
        self.inputs = []

    def add_graph_level(self, name):
        """Add a new depth level to the graph"""

        if name in self._nodes.keys():
            raise KeyError('Graph level with name {} already exists. Existing levels are {}'
                           .format(name, self._nodes.keys()))
        self._nodes[name] = []

    def add_node(self, node, level):
        """Add a node to a given graph level"""

        self._nodes[level].append(node)

    def compute(self, var_dict):
        """Compute result based on given variable dict"""

        for level in reversed(self._nodes.values()):
            for node in level:
                print(node.compute(var_dict))

    def print_graph_levels(self):
        """Print current graph levels"""

        for item in self._nodes.items():
            print(item)

    def serialize_graph(self, file_name):
        """Serialize the graph to pickle format for saving it"""

        # Dump graph object to pickle file
        with open(file_name, 'wb') as outfile:
            pickle.dump(self, outfile)
        outfile.close()

    def show_graph(self):
        """Show the current graph"""

        dot = Digraph()

        # Create nodes
        for level in self._nodes.items():
            with dot.subgraph() as sub:
                sub.attr(rank='same')
                for node in level[1]:
                    sub.node(str(node), type(node).__name__)

        # Create edges
        for level in self._nodes.values():
            for node in level:
                for input_node in node.input_nodes.items():
                    dot.edge(str(input_node[1]), str(node), label=str(input_node[0]))

        dot.graph_attr['rankdir'] = 'BT'
        dot.render('graph.gv', view=True)


def deserialize_graph(file_name):
    """Load graph from a pickle file"""

    # Load graph object from pickle file
    with open(file_name, 'rb') as infile:
        graph = pickle.load(infile)
    infile.close()

    return graph
