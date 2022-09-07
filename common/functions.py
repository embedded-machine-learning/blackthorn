"""Functions"""

import numpy as np

def normalize_ndarray_1d(array: np.ndarray):
    """Normalize a numpy vector to range 0 - 1"""

    if array.max() == array.min():
        return np.zeros(array.shape)

    return (array - array.min()) / (array.max() - array.min())

def add_value_to_ndarray(x, value):
    """Add an value to a numpy array and extend in necessary"""

    if np.count_nonzero(x) >= x.shape[0]:
        x = np.append(x, np.zeros((16), dtype=np.uint32))
    x[np.count_nonzero(x)] = value

    return x
