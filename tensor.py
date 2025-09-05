import torch
from autograd import backward

class tensor:
    def __init__(self, data, requires_grad=False):
        self.data = torch.tensor(data, dtype=torch.float32, requires_grad=False)
        self.grad = None
        self.requires_grad = requires_grad
        self.opb = None
        self.track = None
        self._backward = None

    def __repr__(self):
        return f"tenxar.tensor({self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        other = other if isinstance(other, tensor) else tensor(other)
        result = tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        if result.requires_grad:
            self.track = ('add', (self, other))

        result._backward = backward(self, other, result)
        return result
    
    def __mul__(self, other):
        other = other if isinstance(other, tensor) else tensor(other)
        result = tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        if result.requires_grad:
            self.track = ('mul', (self, other))
        
        result._backward = backward(self, other, result)
        return result

    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * -1

    def __matmul__(self, other):
        other = other if isinstance(other, tensor) else tensor(other)
        result = tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        if result.requires_grad:
            self.track = ('matmul', (self, other))
        
        result._backward = backward(self, other, result)
        return result

    def __rmatmul__(self, other):
        other = other if isinstance(other, tensor) else tensor(other)
        return other @ self
    
    def exp(self):
        x = self.data
        result = tensor(x.exp(), requires_grad=self.requires_grad)
        if result.requires_grad:
            self.track = ('exp', (self))
        result._backward = backward(self, result)
        return result

    def tanh(self):
        x = self.data
        tanh = x.tanh()
        result = tensor(tanh, requires_grad=self.requires_grad)
        if result.requires_grad:
            self.track = ('tanh', (self))
        result._backward = backward(self, tanh, result)
        return result
