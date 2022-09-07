"""Definitions"""

from enum import Enum
from collections import namedtuple

# Enums
class ERROR_MODE(Enum):
    """Error mode enum"""

    THRES = 'thres'
    RATIO = 'ratio'

# Named Tuples
ReturnTuple = namedtuple('ReturnTuple', ['done', 'next_point'])
Point = namedtuple('Point', ['x', 'y'])
