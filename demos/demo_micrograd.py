import numpy as np
from micrograd.engine import Value
from micrograd.nn import MLP as MicrogradMLP
import matplotlib.pyplot as plt
# Generate some random data
np.random.seed(42)
x_data = np.random.randn(100, 3)
y_data = np.random.randn(100, 1)

# Convert data to micrograd Values
def array_to_value(arr):
    return [[Value(x) for x in row] for row in arr]

x_micrograd = array_to_value(x_data)
y_micrograd = array_to_value(y_data)

# Initialize model
nin, nouts = 3, [4, 4, 1]
micrograd_mlp = MicrogradMLP(nin, nouts)

import time
from petitgrad.engine import Tensor
from petitgrad.nn import Layer, MLP

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
def train_step_micrograd(model, x, y, learning_rate):
    # Forward pass
    y_pred = [model(row) for row in x]
    loss = sum((pred - target[0])**2 for pred, target in zip(y_pred, y)) / len(y)
    
    # Backward pass
    model.zero_grad()
    loss.backward()

    # Update parameters
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    
    return loss.data

# Training Micrograd Model
epochs = 1000
learning_rate = 0.01
micrograd_losses = []

start_time = time.time()
for epoch in range(epochs):
    micrograd_loss = train_step_micrograd(micrograd_mlp, x_micrograd, y_micrograd, learning_rate)
    micrograd_losses.append(micrograd_loss)
    
    if epoch % 100 == 0:
        print(f"Micrograd Epoch {epoch}, Loss: {micrograd_loss:.4f}")
micrograd_time = time.time() - start_time


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
    
    # Gradient clipping to prevent explosion
    # max_norm = 1.0
    # total_norm = np.sqrt(sum(np.sum(param.grad**2) for param in model.parameters()))
    # clip_coef = max_norm / (total_norm + 1e-6)
    # if clip_coef < 1:
    #     for param in model.parameters():
    #         param.grad *= clip_coef
    
    # Update parameters
    for param in model.parameters():
        param.data -= learning_rate * param.grad
    
    return loss.data.item()

# Training petitgrad model
petitgrad_losses = []

start_time = time.time()
for epoch in range(epochs):
    petitgrad_loss = train_step_petitgrad(petitgrad_mlp, x_petitgrad, y_petitgrad, learning_rate)
    petitgrad_losses.append(petitgrad_loss)
    
    if epoch % 100 == 0:
        print(f"petitgrad Epoch {epoch}, Loss: {petitgrad_loss:.4f}")
petitgrad_time = time.time() - start_time

# Plot loss comparison
plt.figure(figsize=(10, 6))
plt.plot(petitgrad_losses, label='petitgrad')
plt.plot(micrograd_losses, label='micrograd', linestyle='dashed')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()

# Print training times
print(f"micrograd training time: {micrograd_time:.2f} seconds")
print(f"petitgrad training time: {petitgrad_time:.2f} seconds")
