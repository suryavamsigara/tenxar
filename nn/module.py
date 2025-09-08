from tenxar import tensor
from autograd import no_grad

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True

    def __setattr__(self, name, value):
        """
        The model attributes like weight, bias, layers need to be registered and tracked
        """
        print(f"__setattr__ called: {name} = {type(value)}")

        if name.startswith("_"): # Skipping adding intenal attributes like _parameters, _modules
            super().__setattr__(name, value)
            return
        
        if isinstance(value, Module):
            self._add_module(name, value)

        if isinstance(value, tensor):
            if value.requires_grad:
                print(name, " requires grad")
                self._add_parameter(name, value)
        super().__setattr__(name, value)

    def parameters(self):
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()
        
    def _add_parameter(self, name, param):
        self._parameters[name] = param
        print(name, "-", param, "is added")
    
    def _add_module(self, name, module):
        self._modules[name] = module
        print(name, "-", module, "is added")

    def train(self):
        self.training = True


    def eval(self):
        self.training = False
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        