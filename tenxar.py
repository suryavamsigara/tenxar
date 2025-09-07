import numpy as np
from autograd import (backward_add,
                      backward_matmul,
                      backward_exp,
                      backward_tanh,
                      backward_mean,
                      backward_sum,
                      backward_mul,
                      backward_pow,
                      backward_log)
from typing import Tuple
from autograd import build_computational_order
from autograd import no_grad

__all__ = ['tenxar', 'no_grad']

class tensor:
    def __init__(self, data, requires_grad: bool=False):
        self.data = np.array(data, dtype=np.float32)
        self.grad = None
        self.requires_grad = requires_grad
        self.track = None
        self._backward = lambda : None

    def __repr__(self):
        return f"tenxar.tensor({self.data}, requires_grad={self.requires_grad})"
    
    def _ensure_tensor(self, other):
        return other if isinstance(other, tensor) else tensor(other)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.data.shape)
    
    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def __add__(self, other):
        other = self._ensure_tensor(other)
        result = tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        if result.requires_grad:
            result.track = (self, other)
            result._backward = backward_add(self, other, result)
        return result
    
    def __sub__(self, other):
        other = self._ensure_tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        other = self._ensure_tensor(other)
        return other + (-self)
    
    def __mul__(self, other):
        other = self._ensure_tensor(other)
        result = tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        if result.requires_grad:
            result.track = (self, other)
            result._backward = backward_mul(self, other, result)

        return result

    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * -1

    def __matmul__(self, other):
        other = self._ensure_tensor(other)
        if self.shape[-1] != other.shape[0]:
            raise ValueError(f"Shape mismatch for matmul: {self.shape} and {other.shape}")
        result = tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        if result.requires_grad:
            result.track = (self, other)
            result._backward = backward_matmul(self, other, result)
        return result

    def __rmatmul__(self, other):
        other = self._ensure_tensor(other)
        return other @ self
    
    def __truediv__(self, other):
        other = self._ensure_tensor(other)
        return self * (other ** -1)
    
    def _rtruediv(self, other):
        other = self._ensure_tensor(other)
        return other * (self ** -1)
    
    def exp(self):
        result = tensor(np.exp(self.data), requires_grad=self.requires_grad)
        if result.requires_grad:
            result.track = (self,)
            result._backward = backward_exp(self, result)
        return result
    
    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError(f"Only scalar powers supported")
        result = tensor(self.data ** other, requires_grad=self.requires_grad)
        if result.requires_grad:
            result.track = (self,)
            result._backward = backward_pow(self, other, result)
        return result

    def tanh(self):
        x = self.data
        tanh = np.tanh(x)
        result = tensor(tanh, requires_grad=self.requires_grad)
        if result.requires_grad:
            result.track = (self,)
            result._backward = backward_tanh(self, tanh, result)
        return result
    
    def mean(self, axis=None, keepdims=False):
        result = tensor(self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        if result.requires_grad:
            result._backward = backward_mean(self, result, axis, keepdims)
            result.track = (self,)
        return result
    
    def sum(self, axis=None, keepdims=False):
        result = tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        if result.requires_grad:
            result._backward = backward_sum(self, result, axis, keepdims)
            result.track = (self,)
        return result
    
    def log(self):
        result = tensor(np.log(self.data), requires_grad=self.requires_grad)
        if result.requires_grad:
            result._backward = backward_log(self, result)
            result.track = (self,)
        return result

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)

        for node in build_computational_order(self):
            if node.requires_grad:
                node.grad = np.zeros_like(node.data)

        self.grad = grad

        for node in reversed(build_computational_order(self)):
            node._backward()

    def to_numpy(self) -> np.ndarray:
        return np.array(self.data)
    
    def to_tensor(self):
        if isinstance(self, np.ndarray):
            return tensor(self)
    
    def dtype(self):
        if isinstance(self, tensor):
            return self.data.dtype
        
