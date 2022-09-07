"""Collection of helper functions for model graphs"""

from collections import namedtuple

### Tuples
ConsumerTuple = namedtuple('ConsumerTuple', ['node', 'port'])

### Functions
def param_check(required_params, consumer: ConsumerTuple):
    """Check if a requested consumer port matches its required params"""

    if not isinstance(consumer, ConsumerTuple):
        raise TypeError('Consumer should be of type {}. Is {}'.format(ConsumerTuple, type(consumer)))

    if not consumer.port in required_params:
        raise ValueError('Invalid parameter {} given for {}. Valid parameters are {}.'
                         .format(consumer.port, type(consumer.node).__name__, required_params))



def input_check(required_params, given_params, node):
    """To be removed"""

    if not isinstance(given_params, dict):
        raise ValueError('Params should be a dict. Is {}'.format(type(given_params)))

    if not len(given_params) == len(required_params):
        raise ValueError(('Number of provided parameters does not match required params for {}. Given {}, '
                          'required {}').format(type(node).__name__, list(given_params.keys()), required_params))

    for param in required_params:
        if not param in given_params.keys():
            raise ValueError('Required parameter {} not provided for {}'.format(param, type(node).__name__))
