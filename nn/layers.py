from ..tensor import Tensor
import numpy as np
from .module import Module

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool=True):
        super().__init__()
        limit = np.sqrt(2.0 / in_features)
        weights_data = np.random.uniform(-limit, limit, (in_features, out_features))

        self.weight = Tensor(weights_data, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True) if bias else None

    def forward(self, x):
        result = x @ self.weight
        if self.bias is not None:
            result += self.bias

        return result

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.relu()