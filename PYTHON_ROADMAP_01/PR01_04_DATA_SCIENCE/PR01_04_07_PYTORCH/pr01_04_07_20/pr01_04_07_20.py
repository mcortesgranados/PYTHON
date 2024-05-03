"""
20. Implementing deep learning models for regression tasks to predict continuous-valued outputs.

In regression tasks, the goal is to predict continuous-valued outputs given input data. Deep learning models such as feedforward neural networks (FNNs), 
convolutional neural networks (CNNs), and recurrent neural networks (RNNs) can be used for regression tasks. 
Let's implement an example of a feedforward neural network (FNN) for regression using PyTorch.

In this example:

We generate sample data for regression, consisting of input-output pairs with a linear relationship and added noise.

We define a custom dataset class (RegressionDataset) to load the regression data.

We implement a feedforward neural network (FNN) model for regression using PyTorch's nn.Module interface.

We initialize the model, loss function (Mean Squared Error), and optimizer (Adam).

We create a DataLoader to load the data in batches for training.

We train the FNN model on the training data using gradient descent optimization.

We plot the training loss history to visualize the training progress.

We generate test data for evaluation and predict output values using the trained model.

We plot the true vs predicted values to assess the model's performance.

This example demonstrates how to implement a feedforward neural network (FNN) for regression using PyTorch. 
The model learns to predict continuous-valued outputs given input data, and it can be used for various regression tasks such as predicting house prices, 
stock prices, or time series forecasting.


"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data for regression
np.random.seed(42)
X_train = np.random.rand(100, 1) * 10  # Generate 100 random values between 0 and 10
y_train = 2 * X_train + 1 + np.random.randn(100, 1)  # Linear relationship with noise

# Define a custom dataset for regression
class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# Define a feedforward neural network (FNN) model for regression
class FNNRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FNNRegression, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
input_dim = 1  # Dimensionality of input data
hidden_dim = 32  # Number of hidden units
output_dim = 1  # Dimensionality of output (single value for regression)
model = FNNRegression(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Create DataLoader
train_dataset = RegressionDataset(X_train, y_train)
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Train the model
num_epochs = 100
train_loss_history = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_loss_history.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}')

# Plot training loss
plt.plot(train_loss_history)
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss History')
plt.show()

# Generate test data for evaluation
X_test = np.linspace(0, 10, 100).reshape(-1, 1)  # Generate evenly spaced test data
y_true = 2 * X_test + 1  # True relationship

# Evaluate the model on test data
model.eval()
with torch.no_grad():
    y_pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

# Plot true vs predicted values
plt.plot(X_test, y_true, label='True Values')
plt.plot(X_test, y_pred, label='Predicted Values')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('True vs Predicted Values')
plt.legend()
plt.show()
