import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from petitgrad.engine import Tensor
from petitgrad.nn import Layer, MLP


class TorchMLP(nn.Module):
    def __init__(self, nin, nouts):
        super(TorchMLP, self).__init__()
        sz = [nin] + nouts
        layers = []
        for i in range(len(nouts)):
            layers.append(nn.Linear(sz[i], sz[i+1]))
            if i < len(nouts) - 1:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

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

# Training loop
def train_step_torch(model, x, y, optimizer):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    y_pred = model(x)
    loss = torch.mean((y_pred - y)**2)  # Changed to mean squared error
    
    # Backward pass
    loss.backward()
        
    # Update parameters
    optimizer.step()
    
    return loss.item()

# Training
epochs = 1000
learning_rate = 0.01  # Reduced learning rate
optimizer = torch.optim.Adam(torch_mlp.parameters(), lr=learning_rate)  # Changed to Adam optimizer

torch_losses = []

for epoch in range(epochs):
    torch_loss = train_step_torch(torch_mlp, x_tensor, y_tensor, optimizer)
    torch_losses.append(torch_loss)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {torch_loss:.4f}")


# Print final parameters
print("\nFinal PyTorch MLP parameters:")
for name, param in torch_mlp.named_parameters():
    print(f"Parameter {name}: shape {param.shape}, data:\n{param.data}")

# Evaluate on training data
torch_mlp.eval()
with torch.no_grad():
    final_pred = torch_mlp(x_tensor)
    final_loss = torch.mean((final_pred - y_tensor)**2).item()
    print(f"\nFinal loss on training data: {final_loss:.4f}")


# Initialize model
petitgrad_mlp = MLP(nin, nouts)

# Convert data to petitgrad Tensor
x_petitgrad = Tensor(x_data.reshape(1, *x_data.shape))
y_petitgrad = Tensor(y_data.reshape(1, *y_data.shape))

# Training loop
def train_step_petitgrad(model, x, y, learning_rate):
    """
    Perform a single training step for the petitgrad model.

    Args:
        model (MLP): The petitgrad MLP model.
        x (Tensor): Input tensor.
        y (Tensor): Target tensor.
        learning_rate (float): Learning rate for the optimizer.
    
    Returns:
        float: Loss value for the current step.
    """
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

# Training
petitgrad_losses = []

for epoch in range(epochs):
    petitgrad_loss = train_step_petitgrad(petitgrad_mlp, x_petitgrad, y_petitgrad, learning_rate)
    petitgrad_losses.append(petitgrad_loss)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {petitgrad_loss:.4f}")

# Plot loss
plt.figure(figsize=(10, 6))
plt.plot(petitgrad_losses, label='petitgrad')
plt.plot(torch_losses, label='PyTorch', linestyle='dashed')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()

# Print final parameters
print("\nFinal petitgrad MLP parameters:")
for i, layer in enumerate(petitgrad_mlp.layers):
    print(f"Layer {i} - Weight shape: {layer.weight.data.shape}, Bias shape: {layer.bias.data.shape}")
    print(f"Weight data:\n{layer.weight.data}")
    print(f"Bias data:\n{layer.bias.data}")

# Evaluate on training data
petitgrad_mlp_output = petitgrad_mlp(x_petitgrad)
final_petitgrad_loss = ((petitgrad_mlp_output - y_petitgrad)**2).sum().data.item() / y_petitgrad.data.size
print(f"\nFinal loss on training data (petitgrad MLP): {final_petitgrad_loss:.4f}")
