"""Error policies"""

import numpy as np

def get_policy(policy_type, start_value, stop_value, length, begin_eval=2, begin_decrease=4, end_progression=False):
    """Create a policy"""

    if policy_type == 'std':
        values = np.zeros(length)

        decrease_length = length - begin_decrease + 1
        if end_progression:
            decrease_length = decrease_length - 3
            values[-3] = 1.5
            values[-2] = 1.25
            values[-1] = 1.125
        values[begin_eval:begin_decrease-1] = start_value
        values[begin_decrease-1:begin_decrease-1+decrease_length] = np.linspace(start_value, stop_value, decrease_length)
        return values

    if policy_type == 'const':
        values = np.ones(length) * start_value
        values[0:2] = 0
        values[8:-1] = 10
        values[-1] = 4
        print(values)
        return values

    raise NotImplementedError('Policy of type {} is not implemented'.format(policy_type))
