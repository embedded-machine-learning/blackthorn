"""Some common functions for benchmarking on nvidia jetson platforms"""

import subprocess
import time

import pycuda.driver as cuda
import tensorrt as trt

# Shothanders for model size comparisions
def GiB(val):
    return val * 1 << 30

def MiB(val):
    return val * 1 << 20

class HostDeviceMem():
    """Simple helper data class that's a little nicer to use than a 2-tuple."""

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    """Allocates all buffers required for an engine, i.e. host/device inputs/outputs."""

    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))

        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    """Convenience function for inference"""

    # Event timer
    start_infer = cuda.Event()
    end_infer = cuda.Event()
    start = cuda.Event()
    end = cuda.Event()

    # Transfer input data to the GPU.
    start.record()
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

    # Run inference.
    start_infer.record()
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    #context.execute(batch_size=batch_size, bindings=bindings)
    end_infer.record()

    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    end.record()

    # Synchronize the stream
    stream.synchronize()
    end_infer.synchronize()
    end.synchronize()

    # Get timing
    to_gpu_time = start.time_till(start_infer)
    infer_time = start_infer.time_till(end_infer)
    from_gpu_time = end_infer.time_till(end)

    return (to_gpu_time, infer_time, from_gpu_time)


def check_available_memory(requested_memory):
    """Check if enough memory is available"""

    while cuda.mem_get_info()[0] < (requested_memory + 1000):
        clear_caches()
        if cuda.mem_get_info()[0] < (requested_memory + 1000):
            time.sleep(1)


def clear_caches():
    """Clear caches"""

    try:
        print('Need to clear: {} MB free (of {})'.format(cuda.mem_get_info()[0]/1000000,
                                                         cuda.mem_get_info()[1]/1000000))
        subprocess.run(['sudo', './clear_caches.sh'], check=True)
        print('Cleared: {} MB free (of {})'.format(cuda.mem_get_info()[0]/1000000, cuda.mem_get_info()[1]/1000000))
    except subprocess.CalledProcessError:
        print('Could not clear RAM caches.')
