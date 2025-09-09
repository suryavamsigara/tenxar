from tenxar import tensor
import numpy as np
from nn import Module

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool=True):
        super().__init__()
        self.weight = tensor(np.random.randn(in_features, out_features), requires_grad=True)
        self.bias = tensor(np.zeros(out_features), requires_grad=True) if bias else None

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