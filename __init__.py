from .tenxar import tensor
from .tenxar import to_numpy
import nn

from .autograd import no_grad

__all__ = [
    'tenxar',
    'shape',
    'zero_grad',
    'to_numpy',
    'dtype',
    'no_grad',
    'mean',
    'log'
]