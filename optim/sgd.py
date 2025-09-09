class SGD:
    def __init__(self, parameters, lr=1e-3):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.lr * param.grad
