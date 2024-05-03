"""
19. Developing deep learning models for anomaly detection tasks to identify unusual patterns in data.

Anomaly detection involves identifying unusual patterns or outliers in data. Deep learning models, such as Autoencoders, 
Variational Autoencoders (VAEs), and Generative Adversarial Networks (GANs), can be used for anomaly detection tasks. 
Let's implement an example of using an Autoencoder for anomaly detection using PyTorch.

In this example:

We define a custom dataset for anomaly detection, which includes normal and anomalous data samples.

We create sample data consisting of normal and anomalous data points.

We split the data into training and testing sets and define DataLoaders for both sets.

We define an Autoencoder model for anomaly detection, which compresses input data into a lower-dimensional latent space and reconstructs it.

We train the Autoencoder model to reconstruct normal data and minimize reconstruction error.

We evaluate the trained model on test data and use the reconstruction loss as a measure of anomaly score.

We determine a threshold for anomaly detection based on the reconstruction loss distribution of normal data.

We detect anomalies in the test data based on the threshold and visualize the results.

This example demonstrates how to implement an Autoencoder model for anomaly detection using PyTorch. 
The model learns to reconstruct normal data and identifies anomalies based on high reconstruction errors.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

# Define a custom dataset for anomaly detection
class AnomalyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# Generate sample data
normal_data = np.random.normal(loc=0, scale=1, size=(1000, 2))  # Normal data samples
anomaly_data = np.random.uniform(low=-10, high=10, size=(50, 2))  # Anomalous data samples

# Create labels (0 for normal data, 1 for anomalies)
normal_labels = np.zeros((len(normal_data), 1))
anomaly_labels = np.ones((len(anomaly_data), 1))

# Concatenate normal and anomaly data and shuffle
all_data = np.concatenate((normal_data, anomaly_data), axis=0)
all_labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
idx = np.arange(len(all_data))
np.random.shuffle(idx)
all_data = all_data[idx]
all_labels = all_labels[idx]

# Split data into train and test sets
train_size = int(0.8 * len(all_data))
train_data, test_data = all_data[:train_size], all_data[train_size:]
train_labels, test_labels = all_labels[:train_size], all_labels[train_size:]

# Define DataLoader
train_dataset = AnomalyDataset(train_data, train_labels)
test_dataset = AnomalyDataset(test_data, test_labels)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize model, loss function, and optimizer
input_dim = 2  # Dimensionality of input data
latent_dim = 1  # Dimensionality of latent space
model = Autoencoder(input_dim, latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
train_loss_history = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, _ in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)  # Reconstruction loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_loss_history.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}')

# Function to plot training loss
def plot_training_loss(loss_history):
    plt.plot(loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss History')
    plt.show()

plot_training_loss(train_loss_history)

# Evaluate the model on test data
def evaluate_model(model, dataloader, threshold):
    model.eval()
    with torch.no_grad():
        anomaly_scores = []
        for inputs, _ in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            anomaly_score = torch.mean(loss, dim=1).numpy()  # Average reconstruction loss
            anomaly_scores.extend(anomaly_score)
    anomalies = np.array(anomaly_scores) > threshold
    return anomalies

# Determine threshold using training data
train_outputs = []
for inputs, _ in train_loader:
    outputs = model(inputs)
    loss = criterion(outputs, inputs)
    train_outputs.extend(loss.detach().cpu().numpy())
threshold = np.percentile(train_outputs, 95)  # 95th percentile of reconstruction loss as threshold

# Evaluate model on test data and detect anomalies
anomalies = evaluate_model(model, test_loader, threshold)

# Plot anomalies
plt.scatter(test_data[:, 0], test_data[:, 1], c=anomalies, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Anomaly Detection Results')
plt.colorbar(label='Anomaly')
plt.show()
