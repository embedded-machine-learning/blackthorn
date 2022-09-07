"""Convenience functions to build an engine"""

import datetime
import os
import csv

import numpy as np
import pandas as pd
import tensorrt as trt

import pycuda.autoinit

from nvidia.setup import MAX_WORKSPACE_SIZE_GB
from nvidia.nvidia_common import check_available_memory, GiB
from layers.layer_definitions import ConvolutionTuple

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_convolution_layer(config: ConvolutionTuple):
    """Set up a convolution layer"""

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:

        # Check available memory and set workspace size
        check_available_memory(GiB(MAX_WORKSPACE_SIZE_GB))
        builder.max_workspace_size = GiB(MAX_WORKSPACE_SIZE_GB)

        # Inputs
        input_tensor = network.add_input(name='conv', dtype=trt.float32, shape=(config.d_in, config.w_in, config.h_in))

        # Weights
        conv_weights = np.random.rand(config.d_out, config.d_in, config.kernel_size,
                                      config.kernel_size).astype(np.float32)
        conv_bias = np.random.rand((config.d_out)).astype(np.float32)

        # Network
        conv = network.add_convolution(input=input_tensor, num_output_maps=config.d_out,
                                       kernel_shape=(config.kernel_size, config.kernel_size), kernel=conv_weights,
                                       bias=conv_bias)
        conv.stride = (config.stride, config.stride)
        if config.pad == 'auto':
            conv.padding = (int(config.kernel_size/2), int(config.kernel_size/2))
        else:
            conv.padding = (config.pad, config.pad)

        # Outputs
        network.mark_output(tensor=conv.get_output(0))

        return builder.build_cuda_engine(network)


def build_engine(config):
    """Convenience wrapper for building layer engines"""

    print('Building engine: {} ({})'.format(config, datetime.datetime.now()))

    # Build engine
    if type(config).__name__ == 'ConvolutionTuple':
        engine = build_convolution_layer(config)
    else:
        raise NotImplementedError('Layer of type {} is currently not supported'.format(type(config).__name__))

    return engine

def create_engine(config, path=None):
    """
    Create an engine
    If path is given, the engine is loaded if it exists or built and saved if not.
    If path is None, the engine is just built.
    """

    # Save or load engine if path is given
    if path is not None:
        # Setup path and engine name based on config
        engine_name = '_'.join([str(elem) for elem in config[2:]]) + '.engine'
        engine_dir = type(config).__name__ + '/' + str(config[0]) + '_' + str(config[1])
        engine_name_full = engine_dir + '/' + engine_name

        # Look up requested engine and build if necessary
        df_engine = pd.read_csv(path + '/engine_file.csv')
        if not engine_name in df_engine['name'].values:
            next_engine_id = df_engine.tail(1).iloc[0]['id'] + 1

            # Build engine
            engine = build_engine(config)

            # Update engine file
            with open(path + '/engine_file.csv', mode='a') as engine_file:
                engine_writer = csv.writer(engine_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                engine_writer.writerow([next_engine_id, engine_name_full])

            # Save engine
            print('Saving engine to {}'.format(path + engine_name_full))
            if not os.path.exists(path + engine_dir):
                os.makedirs(path + engine_dir)
            save_engine(engine, path + engine_name_full)
        else:
            # Load engine
            with open(os.path.join(path, engine_name_full), 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
    else:
        engine = build_engine(config)

    return engine

def save_engine(engine, path):
    """Save engine"""

    serialized_engine = engine.serialize()
    with open(path, 'wb') as f:
        f.write(serialized_engine)
        f.close()
    serialized_engine.__del__()
