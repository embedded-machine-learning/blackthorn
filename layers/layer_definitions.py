"""Collection of layer definitions for benchmarking"""

from enum import Enum
from collections import namedtuple

#################################### Enums ####################################

class Convolution(str, Enum):
    """Enum holding configuration options for convolution layer"""

    # Input dimensions
    W_IN = 'w_in'
    H_IN = 'h_in'
    D_IN = 'd_in'

    # Kernel
    D_OUT = 'd_out'
    SIZE = 'kernel_size'
    STRIDE = 'stride'

    # Other
    PAD = 'pad'


############################### Parameter Lists ###############################

CONVOLUTION_PARAMS = [p.value for p in Convolution]

# Dictionary for convenience (in case multiple parameter sets are needed)
LAYER_PARAMETER_DICT = {'Convolution': CONVOLUTION_PARAMS}


################################# Named Tuples ################################

ConvolutionTuple = namedtuple('ConvolutionTuple', CONVOLUTION_PARAMS) # pylint: disable=invalid-name
