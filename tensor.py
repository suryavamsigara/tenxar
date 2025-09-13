import numpy as np
from .autograd import (backward_add,
                      backward_matmul,
                      backward_exp,
                      backward_tanh,
                      backward_sigmoid,
                      backward_relu,
                      backward_mean,
                      backward_sum,
                      backward_mul,
                      backward_pow,
                      backward_log,
                      backward_max,
                      backward_getitem,
                      backward_squeeze,
                      backward_reshape)
from typing import Tuple
from .autograd import build_computational_order
from .autograd import no_grad

__all__ = ['Tensor', 'arange']

class Tensor:
    def __init__(self, data, dtype=np.float32, requires_grad: bool=False):
        self.data = np.array(data, dtype=dtype)
        self.grad = None
        self.requires_grad = requires_grad
        self.track = None
        self.operation = ''
        self._backward = lambda : None

    def __getitem__(self, index):
        if isinstance(index, slice): # for mini batching
            numpy_result = self.data[index]
            return Tensor(numpy_result, requires_grad=False)

        original_index = index
        if isinstance(index, tuple):
            rows, cols = index
            if isinstance(rows, Tensor): rows = rows.to_numpy()
            if isinstance(cols, Tensor): cols = cols.to_numpy()
            original_index = (rows.astype(np.int64), cols.astype(np.int64))
            numpy_result = self.data[original_index]
        else:
            if isinstance(index, Tensor): index = index.to_numpy()
            original_index = index
            numpy_result = self.data[original_index]

        result = Tensor(numpy_result, requires_grad=self.requires_grad)
        if result.requires_grad:
            result.track = (self,)
            result.operation = 'getitem'
            result._backward = backward_getitem(self, original_index, result)
        
        return result

    def __repr__(self):
        return f"tenxar.Tensor({self.data}, requires_grad={self.requires_grad}, dtype={self.dtype})"
    
    def _ensure_Tensor(self, other):
        return other if isinstance(other, Tensor) else Tensor(other)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.data.shape)
    
    @property
    def dtype(self):
        if isinstance(self, Tensor):
            return self.data.dtype
        
    def __len__(self):
        return len(self.data)
    
    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def __add__(self, other):
        other = self._ensure_Tensor(other)
        result = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        if result.requires_grad:
            result.track = (self, other)
            result.operation = 'add'
            result._backward = backward_add(self, other, result)
        return result
    
    def __sub__(self, other):
        other = self._ensure_Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        other = self._ensure_Tensor(other)
        return other + (-self)
    
    def __mul__(self, other):
        other = self._ensure_Tensor(other)
        result = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        if result.requires_grad:
            result.track = (self, other)
            result.operation = 'mul'
            result._backward = backward_mul(self, other, result)

        return result

    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * Tensor(-1, requires_grad=False)

    def __matmul__(self, other):
        other = self._ensure_Tensor(other)
        if self.shape[-1] != other.shape[0]:
            raise ValueError(f"Shape mismatch for matmul: {self.shape} and {other.shape}")
        result = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        if result.requires_grad:
            result.track = (self, other)
            result.operation = 'matmul'
            result._backward = backward_matmul(self, other, result)
        return result

    def __rmatmul__(self, other):
        other = self._ensure_Tensor(other)
        return other @ self
    
    def __truediv__(self, other):
        other = self._ensure_Tensor(other)
        return self * (other ** -1)
    
    def _rtruediv(self, other):
        other = self._ensure_Tensor(other)
        return other * (self ** -1)
    
    def exp(self):
        result = Tensor(np.exp(self.data), requires_grad=self.requires_grad)
        if result.requires_grad:
            result.track = (self,)
            result.operation = 'exp'
            result._backward = backward_exp(self, result)
        return result
    
    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError(f"Only scalar powers supported")
        result = Tensor(self.data ** other, requires_grad=self.requires_grad)
        if result.requires_grad:
            result.track = (self,)
            result.operation = 'pow'
            result._backward = backward_pow(self, other, result)
        return result

    def tanh(self):
        x = self.data
        tanh = np.tanh(x)
        result = Tensor(tanh, requires_grad=self.requires_grad)
        if result.requires_grad:
            result.track = (self,)
            result.operation = 'tanh'
            result._backward = backward_tanh(self, tanh, result)
        return result
    
    def sigmoid(self):
        sig = 1 / (1 + np.exp(-self.data))
        result = Tensor(sig, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result.track = (self,)
            result.operation = 'sigmoid'
            result._backward = backward_sigmoid(self, sig, result)
        return result
    
    def relu(self):
        data = np.maximum(0, self.data)
        result = Tensor(data, requires_grad=self.requires_grad)

        if self.requires_grad:
            result.track = (self,)
            result.operation = 'relu'
            result._backward = backward_relu(self, result)
        return result
    
    def mean(self, axis=None, keepdims=False):
        result = Tensor(self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        if result.requires_grad:
            result._backward = backward_mean(self, result, axis, keepdims)
            result.track = (self,)
            result.operation = 'mean'
        return result
    
    def sum(self, axis=None, keepdims=False):
        result = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        if result.requires_grad:
            result._backward = backward_sum(self, result, axis, keepdims)
            result.track = (self,)
            result.operation = 'sum'
        return result
    
    def log(self):
        result = Tensor(np.log(self.data), requires_grad=self.requires_grad)
        if result.requires_grad:
            result._backward = backward_log(self, result)
            result.track = (self,)
            result.operation = 'log'
        return result
    
    def max(self, axis=None, keepdims=False):
        data = self.data.max(axis=axis, keepdims=keepdims)
        result = Tensor(data, requires_grad=self.requires_grad)

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
    
    def to_Tensor(self):
        if isinstance(self, np.ndarray):
            return Tensor(self)
        
    def arange(*args, dtype=np.int64, requires_grad=False, **kwargs):
        nargs = (arg.to_numpy() if isinstance(arg, Tensor) else arg for arg in args)
        data = np.arange(*nargs, dtype=dtype, **kwargs)
        return Tensor(data, requires_grad=requires_grad)
    
    def squeeze(self, axis=None):
        data = self.data.squeeze(axis=axis)
        result = Tensor(data, requires_grad=self.requires_grad)

        if result.requires_grad:
            result.track = (self,)
            result.operation = 'squeeze'
            result._backward = backward_squeeze(self, result)
        return result
    
    def round(self, decimals=0):
        return Tensor(self.data.round(decimals=decimals), requires_grad=False)
    
    def reshape(self, *shape):
        data = self.data.reshape(*shape)
        result = Tensor(data, requires_grad=self.requires_grad)

        if result.requires_grad:
            result.track = (self,)
            result.operation = 'reshape'
            result._backward = backward_reshape(self, result)
        return result
    
    def rand(*shape, dtype=np.float32, requires_grad=False):
        data = np.random.rand(*shape)
        return Tensor(data, dtype=dtype, requires_grad=requires_grad)
