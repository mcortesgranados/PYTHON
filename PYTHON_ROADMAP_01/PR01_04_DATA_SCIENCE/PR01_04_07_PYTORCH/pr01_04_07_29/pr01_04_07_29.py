"""
29. Training deep learning models for regression tasks with multiple target variables.

Training deep learning models for regression tasks with multiple target variables involves predicting multiple continuous-valued outputs simultaneously. 
This can be achieved by designing a neural network with an appropriate output layer that outputs a vector of predictions corresponding to each target 
variable. In this example, we'll train a simple neural network for a regression task with multiple target variables using PyTorch.

In this example:

We define a simple neural network (RegressionModel) with one hidden layer for regression with multiple target variables.
We generate synthetic data for regression with two target variables: house prices and energy consumption.
We convert the numpy arrays to PyTorch tensors and create a PyTorch dataset and data loader.
We initialize the model, loss function (Mean Squared Error), and optimizer (Adam).
We train the model on the training set for a fixed number of epochs.
After training, we can use the trained model to make predictions for new data.

This example demonstrates how to train a neural network for regression tasks with multiple target variables using PyTorch. 
The same approach can be extended to regression tasks with more than two target variables by adjusting the output size of the neural network's 
final layer accordingly.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Define a simple neural network for regression with multiple target variables
class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Generate synthetic data for regression with multiple target variables
# Example: Predicting house prices and energy consumption based on features like size, location, etc.
num_samples = 1000
num_features = 10
num_targets = 2
X = np.random.randn(num_samples, num_features).astype(np.float32)
# Define two target variables: house prices and energy consumption
y1 = np.random.randn(num_samples, 1).astype(np.float32) * 100000  # House prices
y2 = np.random.randn(num_samples, 1).astype(np.float32) * 1000  # Energy consumption
Y = np.hstack((y1, y2))  # Combine into one array

# Convert numpy arrays to PyTorch tensors
X_tensor = torch.tensor(X)
Y_tensor = torch.tensor(Y)

# Create a PyTorch dataset and data loader
dataset = TensorDataset(X_tensor, Y_tensor)
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
input_size = num_features
output_size = num_targets
model = RegressionModel(input_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}')

# Example of using the trained model to make predictions
# Assuming new_features is a tensor of shape (batch_size, num_features)
new_features = torch.randn(5, num_features)
with torch.no_grad():
    model.eval()
    predictions = model(new_features)
    print("Predictions for new features:")
    print(predictions)
