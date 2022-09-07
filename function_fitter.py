"""The function fitter module"""

import numpy as np
import pandas as pd

from fitter_core import FitterCoreStep, FitterCoreLinear
from common.definitions import ReturnTuple, Point, ERROR_MODE

class Fitter():
    """The function fitter class"""

    def __init__(self, config):

        self.config = config

        # Setup vectors for benchmark data (datapoint: x, measurements: y)
        # ToDo: Change to datapoint
        self.x = np.zeros((self.config['max_iterations']), dtype=np.uint32)
        self.y = np.zeros((self.config['max_iterations']))

        self.fitter_cores = []

        self.next_point = None

        # Some state variables
        self.status = {}
        self.status['iteration'] = 1
        self.iteration = 1
        self.initialized = False
        self.done = False


    def filter_cores(self, model_errors):
        """Filter fitter cores to select best matching function"""

        mask = np.ones(model_errors.shape[0], dtype=bool)
        i = min(self.iteration - 1, len(self.config['selection_error'][ERROR_MODE.RATIO]) - 1)

        if ERROR_MODE.RATIO in self.config['selection_error'].keys():
            if self.config['selection_error'][ERROR_MODE.RATIO][i]:
                mask &= model_errors / model_errors.min() <= self.config['selection_error'][ERROR_MODE.RATIO][i]

        for i, core in enumerate(self.fitter_cores):
            mask[i] &= ~core.ready_to_remove

        for index in sorted(np.where(~mask)[0], reverse=True):
            del self.fitter_cores[index]


    def get_initial_points(self):
        """Get two points to start function fitting"""

        # p = self.config['eval_range'] * self.config['inital_points_thres']
        #return np.array([p, self.config['eval_range'] - p], dtype=np.int)
        return np.array([64, 960], dtype=np.int)


    def get_model(self):
        """Return fitted model"""


    def get_next_point(self, x):
        """Find ideal next evaluation point.

        Required inputs:\n
            - ensemble:
            - x:
        Returns:\n
            - function_values: Functions evaluated at given points x
            - ID of next evaluation point
        """

        # Setup numpy array for scores
        scores = np.zeros([self.config['eval_range']['max'], len(self.fitter_cores)])

        # Calculate scores for all cores
        for i, core in enumerate(self.fitter_cores):
            scores[:, i] = core.get_scores(x)

        # Next point is at maximum score index
        scores = scores.max(axis=1)
        scores[x - 1] = 0
        self.next_point = np.argmax(scores).astype(np.int) + 1

        #print('Next eval point is {}'.format(self.next_point))

        return self.next_point


    def init_fitter(self, x, y):
        """Initialize the fitter with two datapoints"""

        # Test shapes of x and y
        if x.shape[0] != 2 or len(x.shape) > 1:
            raise ValueError('shape mismatch: array x has to be of shape (2, ) - got {}'.format(x.shape))
        if y.shape[0] != 2 or len(y.shape) > 1:
            raise ValueError('shape mismatch: array y has to be of shape (2, ) - got {}'.format(y.shape))

        # Add and initialize fitter cores
        self.fitter_cores = setup_fitter_cores(self.config['cores'], Point(x[0], y[0]), Point(x[1], y[1]))
        if not self.fitter_cores:
            raise ValueError('No filter cores found. Please add at least one core to the config dict.')

        # Add initial samples
        self.x[0:2] = x
        self.y[0:2] = y

        # Find first sample for iteration to start with
        self.get_next_point(self.x[self.x > 0])
        self.initialized = True

        return ReturnTuple(False, self.next_point)


    def iterate(self, x, y):
        """Run a single iteration"""

        if not self.initialized:
            raise RuntimeError('Function fitter not initialized. Please call "get_initial_points" and "init_fitter" ' +
                               'before calling this function.')

        # Do some initial checking
        if y.shape[0] != 1 or len(y.shape) > 1:
            raise ValueError('shape mismatch: array y has to be of shape (parallel_computations, ) - ' +
                             'expected: ({}, ), got: {}'.format(1, y.shape))
        if x != self.next_point:
            print('x does not match suggested next point - expected: {}, got: {}'.format(self.next_point, x))

        # Add new samples
        self.x[self.iteration + 1] = x
        self.y[self.iteration + 1] = y
        #print('Added new point: {}.'.format(Point(x, y[0])))

        model_errors = np.zeros([len(self.fitter_cores)])
        for i, core in enumerate(self.fitter_cores):
            model_errors[i] = core.get_error(self.x[self.x > 0], self.y[self.x > 0], self.iteration)
            if type(core).__name__ == 'FitterCoreLinear':
                model_errors[i] *= 10
        print(model_errors)

        # Do function fitting
        if self.iteration > 1:
            for i, core in enumerate(self.fitter_cores):
                core.fit_model(self.x[self.x > 0], self.y[self.x > 0])

        # Filter fitter cores
        self.filter_cores(model_errors)

        # Fitting is done when:
        #   -) Only one core is left and
        #   -) Only one parameter set inside this core is left and
        #   -) At least 6 iterations are done
        # Fitting is aborted when max_iterations are done
        if self.iteration == self.config['max_iterations']:
            self.fitter_cores = self.fitter_cores[model_errors == model_errors.min()]
            return_tuple = ReturnTuple(True, 0)
        elif len(self.fitter_cores) == 1 and self.fitter_cores[0].param_set_selected and self.iteration >= 6:
            return_tuple = ReturnTuple(True, 0)
        else:
            self.get_next_point(self.x[self.x > 0])
            return_tuple = ReturnTuple(False, self.next_point)

            #print('\nDone after {} iterations'.format(self.iteration))
            #print('Result')
            #print(pd.DataFrame(self.fitter_cores[0].param_ensemble.T))
            #print('Measured Points')
            #print(self.x[self.x > 0])

        self.iteration = self.iteration + 1

        return return_tuple


# Some helper functions
def setup_fitter_cores(core_config, p0, p1):
    """Setup cores for model fitting"""

    fitter_cores = []

    for core in core_config.keys():
        if core == 'fitter_core_step':
            fitter_cores.append(FitterCoreStep(core_config[core], p0, p1))
            print('Added {} with initial points: {} and {}.'.format(core, p0, p1))
        elif core == 'fitter_core_linear':
            fitter_cores.append(FitterCoreLinear(core_config[core], p0, p1))
            print('Added {} with initial points: {} and {}.'.format(core, p0, p1))
        else:
            print('Implementation for core {} not found. Not added.'.format(core))

    return fitter_cores
