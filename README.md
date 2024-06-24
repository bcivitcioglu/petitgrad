# petitgrad: A Matrix-based Autograd Engine

![status](https://img.shields.io/badge/is_it_ready-it_works-green)
![License](https://img.shields.io/badge/Licence-MIT-blue)

<p align="center">
  <img src="imgs/petitgrad_dog.png" alt="petitgrad Logo" width="400"/>
</p>

petitgrad is a matrix-based automatic differentiation engine inspired by [micrograd](https://github.com/karpathy/micrograd), a scalar autograd engine built by [Andrej Karpathy](https://github.com/karpathy). While [micrograd](https://github.com/karpathy/micrograd) operates on scalars, petitgrad attempts to extend this concept to work with matrices. It comes with 3 bites, tested for 3 dimensional tensors, thus the name petitgrad 😄.

> [!IMPORTANT]
> A huge thank you to [Andrej Karpathy](https://github.com/karpathy) for his [micrograd](https://github.com/karpathy/micrograd), and his [amazing educational video](https://youtu.be/VMj-3S1tku0?si=Ij_x0M7_95CrQ8oE). This project was built upon the foundations laid by [micrograd](https://github.com/karpathy/micrograd), attempting to extend its scalar operations to matrix operations with broadcasting (!).

This is purely a personal challenge level project, built for educational purposes and to experience broadcasting implementation challenges.

## Installation

```bash
pip install petitgrad
```

## Features

- Matrix-based automatic differentiation
- Support for basic neural network operations
- Compatible with numpy arrays

> [!CAUTION]
> petitgrad is tested in various cases, but it is the result of one-day coding sprint, and probably has many missing cases.

> [!NOTE]
> But it does work!

## Quick Start

```python
from petitgrad.engine import Tensor
from petitgrad.nn import Layer, MLP
import numpy as np

# Create a simple MLP
model = MLP(3, [4, 4, 1])

# Generate some dummy data
X = Tensor(np.random.randn(10, 3))
y = Tensor(np.random.randn(10, 1))

# Forward pass
y_pred = model(X)

# Compute loss
loss = ((y_pred - y) ** 2).sum()

# Backward pass
loss.backward()

print(f"Loss: {loss.data}")
```

## Examples and comparison benchmarking using torch and micrograd

### Sanity check with torch

Created the same MLP in both petitgrad and torch to make sure on simple tests petitgrad gives reasonable results. Below is a summary, for more detail check the "demos" directory.

```python
# Initialize model
nin, nouts = 3, [4, 4, 1]
torch_mlp = TorchMLP(nin, nouts)

# Generate some random data
np.random.seed(42)
torch.manual_seed(42)
x_data = np.random.randn(100, 3)
y_data = np.random.randn(100, 1)

# Convert data to PyTorch tensors
x_tensor = torch.tensor(x_data, dtype=torch.float32)
y_tensor = torch.tensor(y_data, dtype=torch.float32)

# Fix the training parameters
epochs = 1000
learning_rate = 0.01  # Reduced learning rate
optimizer = torch.optim.Adam(torch_mlp.parameters(), lr=learning_rate)  # Changed to Adam optimizer

# Initialize the petitgrad model
petitgrad_mlp = MLP(nin, nouts)

# Convert data to petitgrad Tensor
x_petitgrad = Tensor(x_data.reshape(1, *x_data.shape))
y_petitgrad = Tensor(y_data.reshape(1, *y_data.shape))
```

When the results are compared, we got the loss function evolution as:

<p align="center">
  <img src="imgs/petitgrad-torch.png" alt="petitgrad-Torch" width="800"/>
</p>

### Sanity and speed check with micrograd

Created the same MLP in both petitgrad and micrograd to make sure petitgrad agrees with micrograd. For more detail check the "demos" directory.

```python

# Initialize petitgrad model

petitgrad_mlp = MLP(nin, nouts)

# Convert data to petitgrad Tensor

x_petitgrad = Tensor(x_data.reshape(1, *x_data.shape))
y_petitgrad = Tensor(y_data.reshape(1, *y_data.shape))

# Training loop for petitgrad model

def train_step_petitgrad(model, x, y, learning_rate):
    for param in model.parameters():
    param.zero_grad()

        # Forward pass
        y_pred = model(x)
        loss = ((y_pred - y)**2).sum() / y.data.size

        # Backward pass
        loss.backward()

    # Update parameters
    for param in model.parameters():
        param.data -= learning_rate * param.grad

    return loss.data.item()

```

We do the same thing using micrograd.

```python
from micrograd.engine import Value
from micrograd.nn import MLP as MicrogradMLP

# Generate some random data

np.random.seed(42)
x_data = np.random.randn(100, 3)
y_data = np.random.randn(100, 1)

# Convert data to micrograd Values

def array_to_value(arr):
    return [[Value(x) for x in row] for row in arr]

x_micrograd = array_to_value(x_data)
y_micrograd = array_to_value(y_data)

# Micrograd Model

nin, nouts = 3, [4, 4, 1]
micrograd_mlp = MicrogradMLP(nin, nouts)

# Training loop for Micrograd

def train_step_micrograd(model, x, y, learning_rate): # Forward pass
    y_pred = [model(row) for row in x]
    loss = sum((pred - target[0])\*\*2 for pred, target in zip(y_pred, y)) / len(y)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Update parameters
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    return loss.data
```

Fix the training parameters.

```python
# Training Micrograd Model

epochs = 1000
learning_rate = 0.01
micrograd_losses = []
```

And finally we time it.

```python
start_time = time.time()
for epoch in range(epochs):
micrograd_loss = train_step_micrograd(micrograd_mlp, x_micrograd, y_micrograd, learning_rate)
micrograd_losses.append(micrograd_loss)

    if epoch % 100 == 0:
        print(f"Micrograd Epoch {epoch}, Loss: {micrograd_loss:.4f}")

micrograd_time = time.time() - start_time

```

and similarly for petitgrad,

```python
start_time = time.time()
for epoch in range(epochs):
    petitgrad_loss = train_step_petitgrad(petitgrad_mlp, x_petitgrad, y_petitgrad, learning_rate)
    petitgrad_losses.append(petitgrad_loss)

    if epoch % 100 == 0:
        print(f"petitgrad Epoch {epoch}, Loss: {petitgrad_loss:.4f}")

petitgrad_time = time.time() - start_time
```

<p align="center">
  <img src="imgs/petitgrad-micrograd.png" alt="petitgrad-Micrograd" width="800"/>
</p>

The output is roughly the same through many trials. The matrix operations make the process significantly faster.

```
micrograd training time: 76.99 seconds
petitgrad training time: 0.42 seconds
```

## Architecture Overview

petitgrad's architecture consists of three main components:

1. **Tensor**: The core data structure that wraps numpy arrays and tracks computational history.
2. **Autograd Engine**: Handles automatic differentiation by building and traversing the computational graph.
3. **Neural Network Modules**: Includes layers and activation functions built on top of the Tensor class.

<p align="center">
  <img src="imgs/petitgrad_architecture.jpeg" alt="petitgrad Architecture" width="400"/>
</p>

## Roadmap

- [ ] Implement more activation functions such as tanh, sigmoid
- [ ] Implement some operations such as mean, std
- [ ] Implement a few cost functions such as cross entropy and mse

## A logo proposal

<p align="center">
  <img src="imgs/petitgrad_logo.png" alt="petitgrad Logo" width="400" style="border-radius: 50%;"/>
</p>

## Contributing

Contributions to petitgrad are welcome! Please feel free to submit a Pull Request. Or, build "centigrad" 😄.

## License

MIT License
