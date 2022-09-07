"""Common functions for the benchmarking tool"""

from enum import Enum
from collections import namedtuple
from scipy.optimize import curve_fit
import numpy as np

import models

class ERROR_MODE(Enum):
    """"""

    THRES = 'thres'
    RATIO = 'ratio'


ReturnTuple = namedtuple('ReturnTuple', ['done', 'next_point'])
EvalRange = namedtuple('EvalRange', ['lower', 'upper'])
Point = namedtuple('Point', ['x', 'y'])

def calculate_and_filter_error(values, y, mode=None):
    """"""

    # Calculate error
    values = values.transpose([1, 0, 2])
    error = np.square(values - y).sum(axis=1)

    # Do filtering
    if mode:
        if ERROR_MODE.RATIO in mode.keys():
            column_min = error.min()
            print('Min error is {}'.format(column_min))
            error[error / column_min > mode[ERROR_MODE.RATIO]] = 0
        if ERROR_MODE.THRES in mode.keys():
            error[error >= mode[ERROR_MODE.THRES]] = 0

    # Add dimension to match function ensemble
    error = error[np.newaxis, :, :]

    return error

def filter_ensemble(ensemble, error):
    """"""

    # Concatinate ensemble and error
    ensemble = np.concatenate((ensemble, error))

    # Remove all functions from ensemble where error was set to 0 for all slices
    sort_index = ensemble[3].argsort(axis=0)
    ensemble = ensemble[:, sort_index, np.arange(ensemble.shape[2])]
    filter_index = np.any(ensemble[3, :, :] != 0, axis=1)
    ensemble = ensemble[:, filter_index, :]

    # Remove all functions from ensemble where step height is close to zero for all slices
    sort_index = ensemble[2].argsort(axis=0)
    ensemble = ensemble[:, sort_index, np.arange(ensemble.shape[2])]
    filter_index = np.any(ensemble[2, :, :] > 1e-10, axis=1)
    #print(filter_index)
    ensemble = ensemble[:, filter_index, :]

    # Remove error from ensemble
    ensemble = np.delete(ensemble, 3, axis=0)

    return ensemble


def eval_step_functions(ensemble, limit=1024):
    """Evaluate ensembles of step function at given points.

    Required inputs:\n
        - ensemble:
    Optional inputs:\n
        - limit: Last point to be evaluated (default 1024)
    Returns:\n
        - function values:
    """

    d = ensemble[0]
    w = ensemble[1]
    h = ensemble[2]

    eval_points = np.arange(0, limit + 1)
    eval_points = eval_points[:, np.newaxis, np.newaxis]

    # Calculate values for all functions at all evaluation points
    function_values = d + np.floor(eval_points / w) * h

    return function_values
