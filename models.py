"""Model Collection"""

import numpy as np

from benchmark_common import Point

# List of available models (class names)
MODEL_TYPES = ['LinearModel', 'StepModel']


# Base class for all models
class BaseModel():
    """Base class for models"""

    @classmethod
    def estimate_params(cls, p0: Point, p1: Point, config):
        """Estimate possible parameters based on two measured points p0 and p1"""

        raise NotImplementedError('Method "estimate_params" not implemented for {}'.format(cls.__name__))

    @classmethod
    def evaluate(cls, params, eval_points):
        """Evaluate model at given point"""

        raise NotImplementedError('Method "evaluate" not implemented for {}'.format(cls.__name__))

    @classmethod
    def fit_function(cls, eval_points):
        """Function used for model fitting"""

        raise NotImplementedError('Method "fit_function" not implemented for {}'.format(cls.__name__))


# Step model
class StepModel(BaseModel):
    """Step Model

    Implements a step model: :math:`f(x)=d+\u230Ax/w\u230B*h`
    """

    @staticmethod
    def estimate_params(p0: Point, p1: Point, config):
        """Estimate possible parameters based on two measured points p0 and p1"""

        # Prepare list of possible width values
        w = np.arange(config['step_width']['min'], config['step_width']['max'] + 1, config['step_width']['interval'])

        # Estimate other parames for each width
        d = (p1.y * np.floor((p0.x - 1) / w) - p0.y * np.floor((p1.x - 1) / w)) /\
            (np.floor((p0.x - 1) / w) - np.floor((p1.x - 1) / w))
        h = (p0.y - p1.y) / (np.floor((p0.x - 1) / w) - np.floor((p1.x - 1) / w))

        return np.stack((d, w, h))

    @staticmethod
    def evaluate(params, eval_points):

        d = params[0]
        w = params[1]
        h = params[2]

        return d + np.floor((eval_points - 1) / w) * h

    @staticmethod
    def fit_function(eval_points, d, w, h):
        """Function for model fitting

        Step model specific: Additional term included to guide the step width to integer values
        """

        return d + np.floor((eval_points - 1) / w) * h + np.square((np.around(w, decimals=0) - w))


# Linear model
class LinearModel(BaseModel):
    """Linear Model

    Implements a linear model: :math:`f(x)=m*x+b`
    """

    @staticmethod
    def estimate_params(p0: Point, p1: Point, config):

        m = np.array((p1.y - p0.y) / (p1.x - p0.x))
        b = np.array((p1.x * p0.y - p0.x * p1.y) / (p1.x - p0.x))

        m = m[np.newaxis]
        b = b[np.newaxis]

        return np.stack((m, b))

    @staticmethod
    def evaluate(params, eval_points):

        m = params[0]
        b = params[1]

        return eval_points * m + b

    @staticmethod
    def fit_function(eval_points, m, b):

        return eval_points * m + b
