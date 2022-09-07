"""Graph test script"""

import os
import sys

sys.path.append(os.path.abspath('..'))

from model_graph import graph, helpers, nodes

my_graph = graph.EstimationGraph()

my_graph.add_graph_level('d_in')
root_node = nodes.StepFunction('d_in', 'root_node')
my_graph.add_node(root_node, 'd_in')

my_graph.add_graph_level('k')
step_d = nodes.StepFunction('k', 'step_d', helpers.ConsumerTuple(root_node, 'd'))
feed_w = nodes.FeedThrough('feed_w', helpers.ConsumerTuple(root_node, 'w'))
step_h = nodes.StepFunction('k', 'step_h', helpers.ConsumerTuple(root_node, 'h'))
my_graph.add_node(step_d, 'k')
my_graph.add_node(feed_w, 'k')
my_graph.add_node(step_h, 'k')

my_graph.add_graph_level('last')
const_d1 = nodes.Constant(2, 'const_d1', helpers.ConsumerTuple(step_d, 'd'))
const_w1 = nodes.Constant(8, 'const_w1', helpers.ConsumerTuple(step_d, 'w'))
const_h1 = nodes.Constant(1, 'const_h1', helpers.ConsumerTuple(step_d, 'h'))
const_feed = nodes.Constant(32, 'const_feed', helpers.ConsumerTuple(feed_w, 'in'))
const_d2 = nodes.Constant(3, 'const_d2', helpers.ConsumerTuple(step_h, 'd'))
const_w2 = nodes.Constant(9, 'const_w2', helpers.ConsumerTuple(step_h, 'w'))
const_h2 = nodes.Constant(2, 'const_h2', helpers.ConsumerTuple(step_h, 'h'))
my_graph.add_node(const_d1, 'last')
my_graph.add_node(const_w1, 'last')
my_graph.add_node(const_h1, 'last')
my_graph.add_node(const_feed, 'last')
my_graph.add_node(const_d2, 'last')
my_graph.add_node(const_w2, 'last')
my_graph.add_node(const_h2, 'last')

#my_graph.print_graph_levels()
var_dict = {'d_in': 33, 'k' : 64}
my_graph.compute(var_dict)
my_graph.show_graph()
