"""Temp file for helpers"""

import numpy as np

from common.definitions import ERROR_MODE
from common.error_policy import get_policy
from function_fitter import Fitter

def get_fitter_cfg():
    """"""

    # Config for models
    step_width_cfg = {'min': 8, 'max': 512, 'interval': 8}
    step_model_cfg = {'step_width': step_width_cfg}
    linear_model_cfg = {}

    # Evaluation range
    eval_range_cfg = {'min': 1, 'max': 1024}

    # Error filtering
    error_cfg = {ERROR_MODE.THRES: 0.025,
                 ERROR_MODE.RATIO: get_policy('std', 20, 2, 16, end_progression=True)}
    selection_cfg = {ERROR_MODE.RATIO: get_policy('std', 5, 2, 16, begin_eval=5, begin_decrease=8)}

    # Cores
    score_params = {'unique': 1,
                    'range': 1,
                    'dist' : 5}
    fitter_core_step_cfg = {'model_cfg': step_model_cfg,
                            'eval_range': eval_range_cfg,
                            'error_mode': error_cfg,
                            'score_params': score_params,
                            'outlier_detection': True,
                            'complexity': 5}
    fitter_core_linear_cfg = {'model_cfg': linear_model_cfg,
                              'eval_range': eval_range_cfg,
                              'error_mode': error_cfg,
                              'outlier_detection': False,
                              'complexity': 1}

    # Fitter config
    fitter_cfg = {}
    fitter_cfg['cores'] = {'fitter_core_step': fitter_core_step_cfg, 'fitter_core_linear': fitter_core_linear_cfg}
    fitter_cfg['eval_range'] = eval_range_cfg
    fitter_cfg['selection_error'] = selection_cfg
    fitter_cfg['max_iterations'] = 48

    return fitter_cfg


def run_inner(backend, fitter_cfg, bench_layer):

    # Setup fitter
    fitter = Fitter(fitter_cfg)

    # Get initial points (only use inference time)
    x = np.array([64, 960])
    y = np.zeros((2))

    bench_layer.update_estimation_value(x[0])
    _, y[0], _ = backend.benchmark_model_avg(bench_layer.get_layer(), 500)
    bench_layer.update_estimation_value(x[1])
    _, y[1], _ = backend.benchmark_model_avg(bench_layer.get_layer(), 500)

    # Init fitter
    result = fitter.init_fitter(x, y)

    # Do fitting
    y = np.zeros(1)
    while not result.done:
        bench_layer.update_estimation_value(result.next_point)
        _, infer_time, _ = backend.benchmark_model_avg(bench_layer.get_layer(), 500)
        y[0] = infer_time
        result = fitter.iterate(result.next_point, y)

    return np.squeeze(fitter.fitter_cores[0].param_ensemble[0:-1])
