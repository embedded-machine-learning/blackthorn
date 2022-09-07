"""Module to collect all nodes for building up the computational graph"""

import numpy as np

from model_graph.helpers import ConsumerTuple, param_check


###################################### Base Class ######################################
class Node:
    """Basic node class"""

    def __init__(self, consumer, required_params=None):

        self.input_nodes = {}
        self.required_params = required_params
        self.consumer_node = None
        self.var_name = None
        self.name = ""

        # Set node as input node for consumer
        if consumer:
            param_check(consumer.node.required_params, consumer)
            self.consumer_node = consumer.node
            self.consumer_node.input_nodes[consumer.port] = self

        self.output = np.NaN

    def compute(self, var_dict=None):
        """Compute the underlying function for this node"""

        raise NotImplementedError('Function "compute" not implemented for {}'.format(type(self).__name__))

    def get_input_params(self):
        """Get input parameters required for this node."""

        input_params = {}
        for param in self.required_params:
            if not param in self.input_nodes.keys():
                raise ValueError('Required parameter {} not provided for {}'.format(param, self))
            if np.isnan(self.input_nodes[param].output):
                raise ValueError('Value for parameter {} not computed. Input node is {} ({})'
                                 .format(param, self.input_nodes[param], self))
            input_params[param] = self.input_nodes[param].output

        return input_params

    def get_var_from_dict(self, var_dict):
        """Get required variable from the provided dict"""

        if not self.var_name in var_dict.keys():
            raise ValueError('Variable {} not found in dictionary ({}).'.format(self.var_name, self))

        return var_dict[self.var_name]


###################################### Implementations ######################################
class Constant(Node):
    """Node with a constant value"""

    def __init__(self, value, node_name: str, consumer: ConsumerTuple = None):

        super().__init__(consumer)
        self.value = value
        self.name = node_name

    def compute(self, var_dict=None):
        """Compute function for Constant node"""

        self.output = self.value
        return self.output


class FeedThrough(Node):
    """Node with a direct feed through (no interaction with variable at this level)"""

    def __init__(self, node_name: str, consumer: ConsumerTuple = None):

        # Set required params
        required_params = ['in']

        super().__init__(consumer, required_params)
        self.name = node_name

    def compute(self, var_dict=None):
        """Compute function for feed through node"""

        input_params = self.get_input_params()

        self.output = input_params['in']
        return self.output


class LinearFunction(Node):
    """Node for calculating a linear function"""

    def __init__(self, var_name: str, node_name: str, consumer: ConsumerTuple = None):

        # Set required params
        required_params = ['m', 'b']

        super().__init__(consumer, required_params)
        self.var_name = var_name
        self.name = node_name

    def compute(self, var_dict=None):
        """Compute function for Step node"""

        input_params = self.get_input_params()
        var = self.get_var_from_dict(var_dict)

        self.output = input_params['m'] * var + input_params['b']
        return self.output


class StepFunction(Node):
    """Node for calculating a step function"""

    def __init__(self, var_name: str, node_name: str, consumer: ConsumerTuple = None):

        # Set required params
        required_params = ['d', 'w', 'h']

        super().__init__(consumer, required_params)
        self.var_name = var_name
        self.name = node_name

    def compute(self, var_dict=None):
        """Compute function for Step node"""

        input_params = self.get_input_params()
        var = self.get_var_from_dict(var_dict)

        self.output = input_params['d'] + np.floor((var - 1) / input_params['w']) * input_params['h']
        return self.output
