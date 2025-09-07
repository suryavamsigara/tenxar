from tenxar import tensor
from autograd import no_grad

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._add_module(name, value)

        if isinstance(value, tensor):
            if value.requires_grad:
                self._add_parameter(name, value)
        super().__setattr__(name, value)

    def parameters(self):
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()
        
    def _add_parameter(self, name, param):
        if not isinstance(param, tensor):
            raise ValueError("Parameters must be a tenxar.tensor")
        self._parameters[name] = param
    
    def _add_module(self, name, module):
        self._modules[name] = module

    
    def train(self):
        self.training = True
        

    def eval(self):
        self.training = False
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        