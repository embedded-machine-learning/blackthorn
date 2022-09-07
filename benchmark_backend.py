"""Common API for running benchmarks on different platforms"""

import importlib

class BackendManager():
    """"""

    def __init__(self, platform):

        # Load backend for given platform
        if platform == 'nvidia':
            self.executer = importlib.import_module('nvidia.engine_executer')
        else:
            raise NotImplementedError('Platform {} is currently not supported'.format(platform))

    def benchmark_model(self, config, path=None):
        """Benchmark a model defined by the given config on a given platform"""

        # Run model
        to_gpu_time, infer_time, from_gpu_time = self.executer.run_model(config, path)

        return (to_gpu_time, infer_time, from_gpu_time)

    def benchmark_model_avg(self, config, nr_runs, path=None):
        """Benchmark a model defined by the given config on a given platform"""

        # Run model
        to_gpu_time, infer_time, from_gpu_time = self.executer.run_model_multiple(config, nr_runs, path)

        return (to_gpu_time, infer_time, from_gpu_time)
