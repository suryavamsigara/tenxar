# TENXAR: A Mini Deep Learning Framework from Scratch

Tenxar is a mini deep learning framework I built from the ground up in Python using only NumPy. Inspired by the API of PyTorch, this project is my implementation of the core components that power modern deep learning, including a dynamic computation graph, a complete autograd engine, and a neural network library.


## Core Features:
* Tensor class: The core data structure of Tenxar. A Tensor wraps around NumPy and stores data, gradients, keeps track of gradients that created it. It integrates with autograd engine for backpropagation.
* Dynamic Computational Graph: Every operation creates a computational graph node linking inputs and outputs. The nodes are topologically sorted for backpropagation. And during `.backward()`, tenxar traverses this graph in reverse, applying the chain rule to compute gradients automatically.
* Automatic Differentiation(Autograd): A powerful engine that computes gradients for any sequence of Tensor operations using backpropagation
* Neural Network Layers: Linear layer, ReLU
* Loss Functions: Cross entropy, binary cross entropy
* Optimizers: Stochastic Gradient Descent (SGD)
* `tenxar.no_grad()`: context manager


## API Overview
`tenxar.Tensor`
The central class of the framework. It's a multi dimensional array
It contains a wide range of differentiable operations (+, *, @, .sigmoid())

`tenxar.nn`
It contains the building blocks for creating neural networks
* `tenxar.nn.Module`: The base class for all neural network modules
* `tenxar.nn.Linear`: Linear layer
* `tenxar.nn.Sequential`: To stack layers in order
* `tenxar.nn.ReLU`: ReLU activation function
* `tenxar.nn.functional`: Contains loss functions - `cross_entropy`, `binary_cross_entropy`

`tenxar.optim`
It has SGD optimizer

## Project Structure

```
tenxar/
|--- __init__.py
|--- tensor.py
|--- autograd.py
|--- nn/
|     |--- __init__.py
|     |--- functional.py
|     |--- layers.py
|     |--- module.py
|--- optim/
      |--- __init__.py
      |--- sgd.py
```


### Examples
```
import tenxar
x = tenxar.Tensor([1, 2, 3], requires_grad=True)
y = tenxar.Tensor([[4, 5, 6], [1, 2, 5]], requires_grad=True)
```


```
from tenxar import nn
class Model(nn.Module):
    def __init__(self, in_features, out_features, hidden_units=8):
        super().__init__()
        self.layer1 = nn.Linear(in_features=in_features, out_features=hidden_units)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(in_features=hidden_units, out_features=hidden_units)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(in_features=hidden_units, out_features=out_features)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        return self.layer3(x)

model_0 = Model(in_features=2,
                    out_features=4,
                    hidden_units=8)
model_4
```


<img width="847" height="611" alt="image" src="https://github.com/user-attachments/assets/2e3b727b-e900-4177-9866-9f0b0264cba4" />

<img width="691" height="737" alt="image" src="https://github.com/user-attachments/assets/24d0e729-cf0b-422c-b3ed-363097573f29" />

