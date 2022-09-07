"""Convenience functions to execute an engine"""

import pandas as pd

import pycuda.autoinit
import tensorrt as trt

from nvidia.engine_builder import create_engine
from nvidia.setup import MAX_MODEL_SIZE_MB
from nvidia.nvidia_common import check_available_memory, do_inference, allocate_buffers, MiB

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def run_model(config, path=None):
    """Run engine with the given config"""

    check_available_memory(MiB(MAX_MODEL_SIZE_MB))

    # Get engine
    engine = create_engine(config, path)

    # Run engine
    to_gpu_time, infer_time, from_gpu_time = run_model_plain(engine)

    engine.__del__()
    return (to_gpu_time, infer_time, from_gpu_time)


def run_model_plain(engine):
    """Run given engine (requires an already loaded or built engine)"""

    check_available_memory(MiB(MAX_MODEL_SIZE_MB))

    # Allocate buffers
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # Do inference
    with engine.create_execution_context() as context:
        to_gpu_time, infer_time, from_gpu_time = do_inference(context, bindings=bindings, inputs=inputs,
                                                              outputs=outputs, stream=stream)

    return (to_gpu_time, infer_time, from_gpu_time)


def run_model_multiple(config, nr_runs, path=None):
    """Convenience function for running an engine multiple times"""

    l_to_gpu = []
    l_infer = []
    l_from_gpu = []

    check_available_memory(MiB(MAX_MODEL_SIZE_MB))

    # Get engine and allocate buffers
    engine = create_engine(config, path)

    for i in range(nr_runs):
        to_gpu_time, infer_time, from_gpu_time = run_model_plain(engine)

        l_to_gpu.append(to_gpu_time)
        l_infer.append(infer_time)
        l_from_gpu.append(from_gpu_time)
        if not i%50 and i > 0:
            print('Done {}/{}'.format(i, nr_runs))

    df = pd.DataFrame(list(zip(l_to_gpu, l_infer, l_from_gpu)), columns=['to_gpu', 'inference', 'from_gpu'])  
    print('Inference time: {}ms'.format(df['inference'].median()))

    engine.__del__()

    return (df['to_gpu'].median(), df['inference'].median(), df['from_gpu'].median())
