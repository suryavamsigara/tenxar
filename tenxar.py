import numpy as np
from autograd import (backward_add,
                      backward_matmul,
                      backward_exp,
                      backward_tanh,
                      backward_mean,
                      backward_sum,
                      backward_mul,
                      backward_pow,
                      backward_log,
                      backward_max,
                      backward_getitem)
from typing import Tuple
from autograd import build_computational_order
from autograd import no_grad

__all__ = ['tensor', 'zero_grad', 'tanh', 'mean', 'log', 'to_numpy', 'arange']

class tensor:
    def __init__(self, data, dtype=np.float32, requires_grad: bool=False):
        self.data = np.array(data, dtype=dtype)
        self.grad = None
        self.requires_grad = requires_grad
        self.track = None
        self.operation = ''
        self._backward = lambda : None

    def __getitem__(self, index):
        original_index = index
        if isinstance(index, tuple):
            rows, cols = index
            if isinstance(rows, tensor): rows = rows.to_numpy()
            if isinstance(cols, tensor): cols = cols.to_numpy()
            original_index = (rows.astype(np.int64), cols.astype(np.int64))
            numpy_result = self.data[original_index]
        else:
            if isinstance(index, tensor): index = index.to_numpy()
            original_index = index
            numpy_result = self.data[original_index]

        result = tensor(numpy_result, requires_grad=self.requires_grad)
        if result.requires_grad:
            result.track = (self,)
            result.operation = 'getitem'
            result._backward = backward_getitem(self, original_index, result)
        
        return result

    def __repr__(self):
        return f"tenxar.tensor({self.data}, requires_grad={self.requires_grad}, dtype={self.dtype})"
    
    def _ensure_tensor(self, other):
        return other if isinstance(other, tensor) else tensor(other)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.data.shape)
    
    @property
    def dtype(self):
        if isinstance(self, tensor):
            return self.data.dtype
    
    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def __add__(self, other):
        other = self._ensure_tensor(other)
        result = tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        if result.requires_grad:
            result.track = (self, other)
            result.operation = 'add'
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
            result.operation = 'mul'
            result._backward = backward_mul(self, other, result)

        return result

    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * tensor(-1, requires_grad=False)

    def __matmul__(self, other):
        other = self._ensure_tensor(other)
        if self.shape[-1] != other.shape[0]:
            raise ValueError(f"Shape mismatch for matmul: {self.shape} and {other.shape}")
        result = tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        if result.requires_grad:
            result.track = (self, other)
            result.operation = 'matmul'
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
            result.operation = 'exp'
            result._backward = backward_exp(self, result)
        return result
    
    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError(f"Only scalar powers supported")
        result = tensor(self.data ** other, requires_grad=self.requires_grad)
        if result.requires_grad:
            result.track = (self,)
            result.operation = 'pow'
            result._backward = backward_pow(self, other, result)
        return result

    def tanh(self):
        x = self.data
        tanh = np.tanh(x)
        result = tensor(tanh, requires_grad=self.requires_grad)
        if result.requires_grad:
            result.track = (self,)
            result.operation = 'tanh'
            result._backward = backward_tanh(self, tanh, result)
        return result
    
    def mean(self, axis=None, keepdims=False):
        result = tensor(self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        if result.requires_grad:
            result._backward = backward_mean(self, result, axis, keepdims)
            result.track = (self,)
            result.operation = 'mean'
        return result
    
    def sum(self, axis=None, keepdims=False):
        result = tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        if result.requires_grad:
            result._backward = backward_sum(self, result, axis, keepdims)
            result.track = (self,)
            result.operation = 'sum'
        return result
    
    def log(self):
        result = tensor(np.log(self.data), requires_grad=self.requires_grad)
        if result.requires_grad:
            result._backward = backward_log(self, result)
            result.track = (self,)
            result.operation = 'log'
        return result
    
    def max(self, axis=None, keepdims=False):
        data = self.data.max(axis=axis, keepdims=keepdims)
        result = tensor(data, requires_grad=self.requires_grad)

        if result.requires_grad:
            result.track = (self,)
            result.operation = 'max'
            result._backward = backward_max(self, result, axis, keepdims)
        return result

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)

        for node in build_computational_order(self):
            if node.requires_grad and node.grad is None:
                node.grad = np.zeros_like(node.data)

        self.grad = grad

        for node in reversed(build_computational_order(self)):
            node._backward()

    def to_numpy(self) -> np.ndarray:
        return np.array(self.data)
    
    def to_tensor(self):
        if isinstance(self, np.ndarray):
            return tensor(self)
    
    @property
    def dtype(self):
        if isinstance(self, tensor):
            return self.data.dtype
        
    def arange(*args, dtype=np.int64, requires_grad=False, **kwargs):
        print(args)
        nargs = (arg.to_numpy() if isinstance(arg, tensor) else arg for arg in args)
        data = np.arange(*nargs, dtype=dtype, **kwargs)
        return tensor(data, requires_grad=requires_grad)
        
arange = tensor.arange