"""
17. Training deep learning models for time series forecasting tasks such as stock price prediction or weather forecasting.

Time series forecasting involves predicting future values based on past observations. 
We can leverage deep learning models, such as Recurrent Neural Networks (RNNs) or Convolutional Neural Networks (CNNs), 
to perform time series forecasting. Let's implement an example of training a Long Short-Term Memory (LSTM) model for stock price prediction using PyTorch.

In this example:

Data Loading: We load stock price data, in this case, Apple Inc. stock prices, from Yahoo Finance.

Data Preprocessing: We preprocess the data by extracting close prices and normalizing them between 0 and 1 using Min-Max scaling.

Dataset Creation: We create a time series dataset for training the LSTM model.

LSTM Model Definition: We define an LSTM model for time series forecasting using PyTorch's nn.Module interface.

Model Training: We train the LSTM model using the time series dataset and optimize it using Adam optimizer.

Future Price Prediction: We define a function to predict future prices using the trained model and visualize the predictions.

This example demonstrates how to implement a simple LSTM model for time series forecasting using PyTorch. 
The model predicts future stock prices based on historical data, and the predictions are visualized for analysis.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load stock price data (e.g., Apple Inc. stock prices)
data = pd.read_csv('https://query1.finance.yahoo.com/v7/finance/download/AAPL?period1=0&period2=9999999999&interval=1d&events=history')

# Extract close prices
prices = data['Close'].values.astype(float).reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
prices_normalized = scaler.fit_transform(prices)

# Function to create time series dataset
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back):
        X.append(data[i:(i+look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

# Hyperparameters
look_back = 20  # Number of time steps to look back for prediction
num_epochs = 100
batch_size = 64
learning_rate = 0.001

# Create time series dataset
X, Y = create_dataset(prices_normalized, look_back)
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Initialize model, loss function, and optimizer
input_dim = 1
hidden_dim = 64
output_dim = 1
num_layers = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
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

# Function to predict future prices
def predict_future_prices(model, data, look_back, num_predictions):
    predictions = []
    for i in range(num_predictions):
        inputs = data[-look_back:].reshape(1, 1, look_back)
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        with torch.no_grad():
            output = model(inputs)
            predictions.append(output.item())
            data = np.append(data, output.item())
    return predictions

# Predict future prices
num_predictions = 30
future_predictions = predict_future_prices(model, prices_normalized[-look_back:], look_back, num_predictions)

# Denormalize predictions
future_predictions_denormalized = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Plot future predictions
plt.plot(prices, label='True Prices')
plt.plot(range(len(prices), len(prices) + num_predictions), future_predictions_denormalized, label='Future Predictions')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using LSTM')
plt.legend()
plt.show()
