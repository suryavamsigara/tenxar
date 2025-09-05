import numpy as np
from autograd import backward_add
from autograd import backward_mul
from autograd import backward_matmul
from autograd import backward_tanh
from autograd import backward_exp
from typing import Tuple
from autograd import build_computational_order

class tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self.requires_grad = requires_grad
        self.track = ()
        self.grad_fn = lambda : None
        self._backward = None

    def __repr__(self):
        return f"tenxar.tensor({self.data}, requires_grad={self.requires_grad})"
    
    def _ensure_tensor(self, other):
        return other if isinstance(other, tensor) else tensor(other)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def __add__(self, other):
        other = self._ensure_tensor(other)
        result = tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        if result.requires_grad:
            self.track = ('add', (self, other))
            result._backward = backward_add(self, other, result)
        return result
    
    def __mul__(self, other):
        other = self._ensure_tensor(other)
        result = tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        if result.requires_grad:
            self.track = ('mul', (self, other))
            result._backward = backward_mul(self, other, result)

        return result

    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * -1

    def __matmul__(self, other):
        other = self._ensure_tensor(other)
        result = tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        if result.requires_grad:
            self.track = ('matmul', (self, other))
            result._backward = backward_matmul(self, other, result)
        return result

    def __rmatmul__(self, other):
        other = self._ensure_tensor(other)
        return other @ self
    
    def exp(self):
        result = tensor(np.exp(self.data), requires_grad=self.requires_grad)
        if result.requires_grad:
            self.track = ('exp', (self))
            result._backward = backward_exp(self, result)
        return result

    def tanh(self):
        x = self.data
        tanh = np.tanh(x)
        result = tensor(tanh, requires_grad=self.requires_grad)
        if result.requires_grad:
            self.track = ('tanh', (self))
            result._backward = backward_tanh(self, tanh, result)
        return result

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)
        self.grad = grad

        for node in reversed(build_computational_order(self)):
            node._backward()