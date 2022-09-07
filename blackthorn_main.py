"""Main"""

import blackthorn_helpers
from benchmark_backend import BackendManager
from layers.benchmark_layer import BenchmarkLayer
from layers.layer_definitions import Convolution, ConvolutionTuple

if __name__ == '__main__':

    # Select hardware platform
    backend = BackendManager('nvidia')

    # Get fitter config
    fitter_cfg = blackthorn_helpers.get_fitter_cfg()

    # Setup fitter structure
    fitter_inputs = [Convolution.D_IN, Convolution.D_OUT]

    # Init logging

    # Template dict
    conv_layer = BenchmarkLayer(Convolution)
    conv_layer.update_config(ConvolutionTuple(64, 64, 32, 32, 3, 1, 'auto'))
    # Main loop
    for level, fitter_in in enumerate(fitter_inputs):

        if level == 0:
            conv_layer.set_estimation_param(fitter_in)
            result = blackthorn_helpers.run_inner(backend, fitter_cfg, conv_layer)
            print(result)

        #while fitters_done.count(False) > 0:
