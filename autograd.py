import numpy as np
from typing import List, Set
from contextlib import contextmanager

def match_shape(grad, shape):
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] > 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

def backward_getitem(self, index, result):
    def _backward():
        if not _no_grad_mode:
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                np.add.at(self.grad, index, result.grad)
    return _backward

def backward_add(self, other, result):
    def _backward():
        if not _no_grad_mode:
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += match_shape(result.grad, self.data.shape)
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += match_shape(result.grad, other.data.shape)
    return _backward

def backward_mul(self, other, result):
    def _backward():
        if not _no_grad_mode:
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += match_shape(other.data * result.grad, self.data.shape)
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += match_shape(self.data * result.grad, other.data.shape)
    return _backward

def backward_matmul(self, other, result):
    assert self.shape[1] == other.shape[0]; "shape should match"
    def _backward():
        if not _no_grad_mode:
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += result.grad @ other.data.T
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += self.data.T @ result.grad
    return _backward

def backward_tanh(self, tanh, result):
    def _backward():
        if not _no_grad_mode:
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += result.grad * (1 - tanh ** 2)
    return _backward

def backward_sigmoid(self, sig, result):
    def _backward():
        if not _no_grad_mode:
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += result.grad * (sig * (1 - sig))
    return _backward

def backward_exp(self, result):
    def _backward():
        if not _no_grad_mode:
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += result.grad * np.exp(self.data)
    return _backward

def backward_pow(self, other, result):
    def _backward():
        if not _no_grad_mode:
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += result.grad * (other * (self.data ** (other - 1)))
    return _backward

def backward_relu(self, result):
    def _backward():
        if not _no_grad_mode:
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                mask = (self.data > 0)
                self.grad += result.grad * mask
    return _backward

def backward_mean(self, result, axis=None, keepdims=False):
    def _backward():
        if not _no_grad_mode:
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                grad = result.grad
                if axis is None:
                    grad = grad * np.ones_like(self.data) / self.data.size
                else:
                    n = self.data.shape[axis] if isinstance(axis, int) else np.prod([self.data.shape[i] for i in axis])
                    grad = np.expand_dims(grad, axis) if not keepdims else grad
                    grad = grad * np.ones_like(self.data) / n
                self.grad += grad
    return _backward

def backward_sum(self, result, axis=None, keepdims=False):
    def _backward():
        if not _no_grad_mode:
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                grad = result.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis)
                self.grad += np.ones_like(self.data) * grad
    return _backward

def backward_log(self, result):
    def _backward():
        if not _no_grad_mode:
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                grad = result.grad
                grad = grad * np.ones_like(self.data) / self.data
                self.grad += grad
    return _backward

def backward_max(self, result, axis, keepdims):
    def _backward():
        if not _no_grad_mode:
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                max_values = result.data
                grad_incoming = result.grad

                if axis is not None and not keepdims:
                    max_values = np.expand_dims(max_values, axis)
                    grad_incoming = np.expand_dims(grad_incoming, axis)

                mask = (self.data == max_values)

                divisor = mask.sum(axis=axis, keepdims=True)
                divisor[divisor == 0] = 1 # to avoid division by zero

                self.grad += (mask / divisor) * grad_incoming
    return _backward

def backward_squeeze(self, result):
    def _backward():
        if not _no_grad_mode:
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # Reshaping the gradient
                self.grad += np.reshape(result.grad, self.data.shape)
    return _backward

def backward_reshape(self, result):
    def _backward():
        if not _no_grad_mode:
            if self.requires_grad:
                if self.requires_grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += np.reshape(result.grad, self.data.shape)
    return _backward

def build_computational_order(root) -> List:
    computation_order: List = []
    visited: Set = set()

    def build(node):
        if node not in visited:
            visited.add(node)
            if node.track:
                for child in node.track:
                    build(child)
            computation_order.append(node)

    build(root)
    return computation_order

_no_grad_mode = False

@contextmanager
def no_grad():
    global _no_grad_mode
    original = _no_grad_mode
    _no_grad_mode = True
    try:
        yield
    finally:
        _no_grad_mode = original