
"""Fitter Core"""

import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d
from scipy.optimize import curve_fit

import models
from common.definitions import Point, ERROR_MODE
from common.functions import normalize_ndarray_1d

###################################### Base Class ######################################
class FitterCoreBase():
    """Base class for fitter cores"""

    def __init__(self, model, config, p0: Point, p1: Point):


        self.model = model
        self.config = config

        # Get an initial guess for the function params. An empty last column is added to store errors
        param_ensemble = model.estimate_params(p0, p1, config['model_cfg'])
        self.param_ensemble = np.vstack((param_ensemble, np.zeros(param_ensemble.shape[1])))

        # Linear fitter has only one param set
        if self.param_ensemble.shape[1] == 1:
            self.param_set_selected = True
        else:
            self.param_set_selected = False

        self.function_values = None
        self.outliers = np.zeros(2, dtype=bool)
        self.ready_to_remove = False

    def evaluate(self):
        """Evaluate function ensemble"""

        eval_points = np.arange(self.config['eval_range']['min'], self.config['eval_range']['max'] + 1)
        eval_points = eval_points[:, np.newaxis]

        self.function_values = self.model.evaluate(self.param_ensemble, eval_points)

    def find_outliers(self, x, y, iteration):
        """Find outliers based on the median error"""

        if iteration == 1 or not self.config['outlier_detection']:
            self.outliers = np.zeros(x.shape, dtype=bool)
        else:
            # Calculate error for each parameter set at each point (x, y)
            error = np.square(self.function_values[x - 1].T - y)
            # Absolute distance to the median error
            error_distance = np.abs(error.T - np.median(error, axis=1))
            if error_distance.all() == 0:
                error_distance = np.ones(error_distance.shape)
            # Scale distance by its median
            error_distance = error_distance / np.median(error_distance, axis=0)
            # Outliers are points with a high error distance for more than half of all parameter sets
            self.outliers = (error_distance > 10).sum(axis=1) > error_distance.shape[1] / 2
        #print('Outliers found: {}'.format(x[self.outliers]))

    def fit_model(self, x, y):
        """Fit model to given data"""

        raise NotImplementedError('Function "fit_model" not implemented for {}'.format(type(self).__name__))


    def get_error(self, x, y, iteration):
        """Calculate the current error of the model.
        For ensemble models: Return the error of the best fit.
        """

        # Recalculate errors
        # ToDo: Evaluate better scaling options (especially for different benchmark ranges)
        scale = ((self.function_values[x[1] - 1] - self.function_values[x[0] - 1]) / 2)**2
        if scale.any() == 0:
            scale = (self.function_values[x[1] - 1] / 2)**2
        self.find_outliers(x, y, iteration)
        self.param_ensemble[-1] = np.square(self.function_values[x[~self.outliers] - 1].T - y[~self.outliers]).sum(axis=1) / scale

        i = min(iteration - 1, len(self.config['error_mode'][ERROR_MODE.RATIO]) - 1)

        # Mark and remove function choices with high error from ensamble
        mask = np.ones(self.param_ensemble[-1].shape, dtype=bool)
        if not self.param_set_selected:
            error_min = self.param_ensemble[-1].min()
            if ERROR_MODE.THRES in self.config['error_mode'].keys():
                mask &= (self.param_ensemble[-1] <= self.config['error_mode'][ERROR_MODE.THRES]) |\
                        (self.param_ensemble[-1] == error_min)
            if ERROR_MODE.RATIO in self.config['error_mode'].keys():
                if self.config['error_mode'][ERROR_MODE.RATIO][i] > 0:
                    mask &= self.param_ensemble[-1] / error_min <= self.config['error_mode'][ERROR_MODE.RATIO][i]
            # Apply mask
            self.param_ensemble = self.param_ensemble[:, mask]

        if self.param_ensemble.shape[1] == 1:
            self.param_set_selected = True

        return self.param_ensemble[-1].min()

    def get_scores(self, x):
        """Get scores for each possible next point"""

        raise NotImplementedError('Function "get_scores" not implemented for {}'.format(type(self).__name__))


###################################### Implementations ######################################
class FitterCoreStep(FitterCoreBase):
    """Implementation of a step fitter core"""

    def __init__(self, p0: Point, p1: Point, config):
        super().__init__(models.StepModel, p0, p1, config)

    def fit_model(self, x, y):

        for i in range(self.param_ensemble.shape[1]):
            try:
                popt, _ = curve_fit(self.model.fit_function, x[~self.outliers], y[~self.outliers],
                                    p0=[self.param_ensemble[0, i], self.param_ensemble[1, i], self.param_ensemble[2, i]],
                                    method='trf')
                # Mark functions without integer step width...
                if abs(popt[1] - self.param_ensemble[1, i]) > 0.25:
                    if self.param_set_selected:
                        print('Warning: {} has found an invalid fit. Keeping old params'.format(type(self).__name__))
                    else:
                        self.param_ensemble[-1, i] = np.NaN
                        self.param_ensemble[1, i] = popt[1] # Just for debugging
                self.param_ensemble[0, i] = popt[0]
                self.param_ensemble[2, i] = popt[2]
            except RuntimeError:
                print('Optimal params not found. Keeping old params')

        df = pd.DataFrame(self.param_ensemble.T)
        print(df.sort_values(by=df.columns[-1]))
        # ...and remove them
        self.param_ensemble = self.param_ensemble[:, ~np.isnan(self.param_ensemble[-1])]
        self.param_ensemble = self.param_ensemble[:, self.param_ensemble[2] > 1e-10]

        if self.param_ensemble.size == 0:
            self.ready_to_remove = True
        elif self.param_ensemble.shape[1] == 1:
            self.param_set_selected = True

    def get_scores(self, x):

        self.evaluate()

        # Find number of unique values for each eval point mapped to a range 0 - 1
        scaled = (self.function_values.T / self.function_values.max(axis=1)).T
        scores_unique = pd.DataFrame(np.around(scaled.T, decimals=3)).nunique().to_numpy()
        scores_unique = normalize_ndarray_1d(scores_unique)

        # Find value range for each eval point
        scores_range = self.function_values.max(axis=1) - self.function_values.min(axis=1)
        scores_range = normalize_ndarray_1d(scores_range)

        # Find point which is farthest away from all other points
        scores_loc = get_location_scores(x, self.config['eval_range']['max'])

        # Compute scores
        scores = normalize_ndarray_1d(self.config['score_params']['unique'] * scores_unique +
                                      self.config['score_params']['range'] * scores_range +
                                      self.config['score_params']['dist'] * scores_loc)

        return scores + self.config['complexity']


class FitterCoreLinear(FitterCoreBase):
    """Implementation of a linear fitter core"""

    def __init__(self, p0: Point, p1: Point, config):
        super().__init__(models.LinearModel, p0, p1, config)

    def fit_model(self, x, y):

        self.param_ensemble[0:-1, 0], _ = curve_fit(self.model.fit_function, x, y)

    def get_scores(self, x):

        self.evaluate()

        # Find point which is farthest away from all other points
        scores = get_location_scores(x, self.config['eval_range']['max'])

        return scores + self.config['complexity']


def get_location_scores(x, nr_eval_points):
    """Get normalized scores for each possible point based on distances between already benchmarked points
    Default gaussian filter is truncated after 4 standard deviations. Thus sigma is chosen as 1/8 times
    the maximum difference.
    """

    scores = np.zeros([nr_eval_points])
    scores[x - 1] = 1
    sigma = np.diff(np.sort(x)).max() / 8
    scores = gaussian_filter1d(scores, sigma=sigma, mode='constant', cval=0)
    scores = 1 - normalize_ndarray_1d(scores)

    return scores
