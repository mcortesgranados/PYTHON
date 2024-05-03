"""
03. Constructing recurrent neural networks (RNNs) for sequential data analysis such as time series forecasting or natural language processing.

Constructing recurrent neural networks (RNNs) for sequential data analysis, such as time series forecasting or natural language processing, 
involves defining an RNN architecture, preparing the dataset, and training the model. In this example, we'll build a simple 
RNN for time series forecasting using PyTorch.

Explanation:

We import the necessary libraries, including PyTorch.

We define the RNN architecture as a class RNN that inherits from nn.Module. This simple RNN consists of an RNN layer followed by a fully connected layer.

We prepare the dataset by generating synthetic time series data (sine waves) using the generate_data function.

We specify the hyperparameters such as input size, hidden size, output size, sequence length, number of samples, number of epochs, and learning rate.

We initialize the RNN model, define the loss function (MSELoss), and choose an optimizer (Adam) for training.

We iterate through the dataset for a specified number of epochs, compute the predictions of the RNN, calculate the loss, perform backpropagation, 
and update the model parameters using the optimizer.

During training, we print the loss every 10 epochs.

After training, we print "Finished Training" to indicate that the training process is complete.

This example demonstrates how to construct an RNN for time series forecasting using PyTorch. RNNs are powerful models for sequential data analysis, 
and with PyTorch, you have the flexibility to customize and train RNN architectures for various tasks such as time series forecasting, 
natural language processing, and more.

"""

# Importing necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the recurrent neural network (RNN) architecture
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)  # Initial hidden state
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out

# Prepare the dataset (toy example of a sine wave)
def generate_data(seq_length, num_samples):
    data = np.zeros((num_samples, seq_length, 1))
    for i in range(num_samples):
        start = np.random.randint(0, 100)
        end = start + seq_length
        data[i, :, 0] = np.sin(np.linspace(start, end, seq_length))
    return data

# Hyperparameters
input_size = 1  # Input dimension (number of features)
hidden_size = 32  # Hidden layer size
output_size = 1  # Output dimension
seq_length = 10  # Length of input sequences
num_samples = 1000  # Number of training samples
num_epochs = 100  # Number of epochs
learning_rate = 0.001  # Learning rate

# Generate synthetic time series data
data = generate_data(seq_length, num_samples)
inputs = torch.from_numpy(data[:, :-1, :]).float()  # Input sequences (exclude last time step)
labels = torch.from_numpy(data[:, -1, :]).float()  # Target values (next time step)

# Initialize the RNN, loss function, and optimizer
model = RNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the RNN
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Finished Training')
