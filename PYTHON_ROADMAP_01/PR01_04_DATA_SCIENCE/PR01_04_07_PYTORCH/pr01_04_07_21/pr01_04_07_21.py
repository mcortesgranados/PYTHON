"""
21. Training deep learning models for multi-class classification tasks with more than two classes.

Multi-class classification tasks involve predicting one of multiple classes for each input sample. 
Deep learning models such as feedforward neural networks (FNNs), convolutional neural networks (CNNs), 
and recurrent neural networks (RNNs) can be used for multi-class classification. Let's implement an example of training an FNN for a multi-class classification 
task using PyTorch.

In this example:

We generate sample data for a multi-class classification task, where each class has a different distribution in the input space.
We define a custom dataset class (MulticlassDataset) to load the multi-class classification data.
We implement a feedforward neural network (FNN) model (FNNMulticlass) for multi-class classification using PyTorch's nn.Module interface.
We initialize the model, loss function (Cross-Entropy Loss), and optimizer (Adam).
We create a DataLoader to load the data in batches for training.
We train the FNN model on the training data using gradient descent optimization.
We plot the training loss history to visualize the training progress.
We evaluate the trained model on test data and calculate the accuracy of the predictions.

This example demonstrates how to implement a feedforward neural network (FNN) for multi-class classification using PyTorch. 
The model learns to predict one of multiple classes for each input sample, and it can be used for various multi-class classification 
tasks such as image classification, text classification, or medical diagnosis.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data for multi-class classification
np.random.seed(42)
num_samples_per_class = 100
num_classes = 3
X_train = np.zeros((num_samples_per_class * num_classes, 2))
y_train = np.zeros((num_samples_per_class * num_classes,), dtype=int)
for i in range(num_classes):
    ix = range(num_samples_per_class * i, num_samples_per_class * (i + 1))
    r = np.linspace(0.0, 1, num_samples_per_class)  # Radius
    t = np.linspace(i * 4, (i + 1) * 4, num_samples_per_class) + np.random.randn(num_samples_per_class) * 0.2
    X_train[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y_train[ix] = i

# Define a custom dataset for multi-class classification
class MulticlassDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# Define a feedforward neural network (FNN) model for multi-class classification
class FNNMulticlass(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FNNMulticlass, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
input_dim = 2  # Dimensionality of input data
hidden_dim = 32  # Number of hidden units
output_dim = num_classes  # Number of classes (equal to number of output units)
model = FNNMulticlass(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Create DataLoader
train_dataset = MulticlassDataset(X_train, y_train)
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
X_test, y_test = X_train, y_train  # Use training data for simplicity

# Evaluate the model on test data
model.eval()
with torch.no_grad():
    outputs = model(torch.tensor(X_test, dtype=torch.float32))
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == torch.tensor(y_test, dtype=torch.long)).sum().item() / len(y_test)
print(f'Accuracy on test data: {accuracy:.2f}')
