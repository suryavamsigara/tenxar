import numpy as np
from typing import List, Set

def zero_grad(self):
    if self.grad is not None:
        self.grad.fill(0)

def match_shape(grad, shape):
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=0, keepdims=True) # returns a 2D array 
    return grad

def backward_add(self, other, result):
    def _backward():
        if self.requires_grad:
            self.grad += match_shape(result.grad, self.data.shape)
        if other.requires_grad:
            other.grad += match_shape(result.grad, other.data.shape)
    return _backward

def backward_matmul(self, other, result):
    assert self.shape[1] == other.shape[0]; "shape should match"
    def _backward():
        if self.requires_grad:
            self.grad += match_shape(other.data * result.grad, self.data.shape)
        if other.requires_grad:
            other.grad += match_shape(self.data * result.grad, other.data.shape)
    return _backward

def backward_tanh(self, tanh, result):
    def _backward():
        if self.requires_grad:
            self.grad += result.grad * (1 - tanh ** 2)
    return _backward

def backward_exp(self, result):
    def _backward():
        if self.requires_grad:
            self.grad += result.grad * np.exp(self.data)
    return _backward        

computation_order = []
visited = set()

def build_computational_order(root):
    computation_order: List = []
    visited: Set = set()

    def build(node):
        if node not in visited:
            visited.add(node)
            for child in node.track[1]:
                build(child)
            computation_order.append(node)

    build(root)
    return computation_order