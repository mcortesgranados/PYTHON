"""
13. Implementing recurrent neural networks with long short-term memory (LSTM) cells for time series prediction and sequence modeling.

Recurrent Neural Networks (RNNs) with Long Short-Term Memory (LSTM) cells are widely used for time series prediction and sequence 
modeling tasks due to their ability to capture long-term dependencies in sequential data. Let's implement an LSTM-based model in 
PyTorch for time series prediction using the popular Sunspot dataset.

In this example:

Imports: Import necessary libraries including PyTorch, pandas, numpy, and matplotlib.
Load Sunspot Dataset: Load the monthly sunspot dataset from a CSV file and normalize it.
Create Time Series Dataset: Create a time series dataset for LSTM input.
Define LSTM Model: Define a simple LSTM model using PyTorch.
Initialize Model, Loss Function, and Optimizer: Initialize the LSTM model, Mean Squared Error (MSE) loss function, and Adam optimizer.
Train the Model: Train the LSTM model using the time series dataset.
Plot Training Loss: Plot the training loss history.
Predictions: Make predictions using the trained LSTM model.
Denormalize Predictions and True Values: Denormalize the predictions and true values.
Calculate RMSE: Calculate the Root Mean Squared Error (RMSE) between the true and predicted values.
Plot Predictions vs True Values: Visualize the original predictions, LSTM predictions, and true values.

This example demonstrates how to implement an LSTM-based model in PyTorch for time series prediction using the Sunspot dataset. 
The LSTM model is trained to predict future sunspot activity based on historical data and evaluated using RMSE.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load Sunspot dataset
sunspot_data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv')
sunspot_values = sunspot_data['Sunspots'].values.astype(float)

# Normalize dataset
scaler = MinMaxScaler(feature_range=(0, 1))
sunspot_normalized = scaler.fit_transform(sunspot_values.reshape(-1, 1))

# Function to create time series dataset
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back):
        X.append(data[i:(i+look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

# Hyperparameters
look_back = 12  # Number of time steps to look back for prediction
num_epochs = 100
batch_size = 64
learning_rate = 0.001

# Create time series dataset
X, Y = create_dataset(sunspot_normalized, look_back)
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Initialize model, loss function, and optimizer
input_size = 1
hidden_size = 64
num_layers = 2
output_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to train the model
def train_model(model, X, Y, criterion, optimizer, num_epochs, batch_size):
    train_loss_history = []
    for epoch in range(num_epochs):
        for i in range(0, len(X), batch_size):
            inputs = X[i:i+batch_size].to(device)
            targets = Y[i:i+batch_size].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()

        train_loss_history.append(loss.item())
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')
    return train_loss_history

# Train the model
train_loss_history = train_model(model, X_tensor, Y_tensor, criterion, optimizer, num_epochs, batch_size)

# Plot training loss
plt.plot(train_loss_history)
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss History')
plt.show()

# Predictions
model.eval()
with torch.no_grad():
    test_inputs = X_tensor.to(device)
    test_outputs = model(test_inputs).cpu().numpy()

# Denormalize predictions and true values
test_outputs_denormalized = scaler.inverse_transform(test_outputs)
true_values_denormalized = scaler.inverse_transform(Y_tensor.numpy().reshape(-1, 1))

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(true_values_denormalized, test_outputs_denormalized))
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

# Plot predictions vs true values
plt.figure(figsize=(10, 6))
plt.plot(sunspot_values, label='True Values')
plt.plot(np.arange(look_back, len(true_values_denormalized)+look_back), true_values_denormalized, label='Original Predictions')
plt.plot(np.arange(look_back, len(test_outputs_denormalized)+look_back), test_outputs_denormalized, label='LSTM Predictions')
plt.xlabel('Time')
plt.ylabel('Sunspots')
plt.title('Sunspot Time Series Prediction with LSTM')
plt.legend()
plt.show()
