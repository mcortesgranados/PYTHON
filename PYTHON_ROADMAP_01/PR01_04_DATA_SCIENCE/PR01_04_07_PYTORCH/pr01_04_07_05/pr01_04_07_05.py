"""
05. Implementing autoencoders for unsupervised learning and dimensionality reduction.

Certainly! Autoencoders are neural networks that aim to learn efficient representations of data by reconstructing it from a compressed version. 
They consist of two main parts: an encoder that compresses the input data into a latent-space representation, and a decoder that reconstructs 
the original input from this representation. Let's create a simple example of an autoencoder for image reconstruction using PyTorch.

This code implements a simple autoencoder using PyTorch and trains it on the MNIST dataset for image reconstruction. Here's a breakdown of the code:

Imports: Import necessary libraries including PyTorch and torchvision.
Device Configuration: Set the device to GPU if available, else CPU.
Hyperparameters: Define hyperparameters such as image size, hidden size, etc.
Data Loading: Load the MNIST dataset and create a data loader.
Encoder and Decoder Networks: Define the encoder and decoder neural networks.
Move Models to Device: Move the networks to the selected device (CPU or GPU).
Loss Function and Optimizer: Define Mean Squared Error (MSE) loss and Adam optimizer.
Training Function: Define a function to train the autoencoder.
Training Loop: Loop through the dataset and train the autoencoder.
Visualization: Visualize some reconstructed images to see how well the autoencoder is performing.

This example demonstrates how to implement and train an autoencoder for unsupervised learning and dimensionality reduction using PyTorch. 
The trained autoencoder can be used for various tasks such as denoising images, anomaly detection, and feature extraction.

"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device configuration (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
image_size = 784  # 28x28 images from MNIST dataset
hidden_size = 256
latent_size = 64
num_epochs = 20
batch_size = 100
learning_rate = 0.001

# Image processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transform,
                                           download=True)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

# Encoder network
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(image_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, latent_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Decoder network
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, image_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Instantiate encoder and decoder
encoder = Encoder().to(device)
decoder = Decoder().to(device)

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

# Function to flatten image tensor
def flatten(x):
    return x.view(x.size(0), -1)

# Function to train the autoencoder
def train_autoencoder(model, dataloader):
    total_loss = 0.0
    for i, (images, _) in enumerate(dataloader):
        images = images.to(device)
        images = flatten(images)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, images)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Training the autoencoder
for epoch in range(num_epochs):
    loss = train_autoencoder(encoder, train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}")

# Visualize some reconstructed images
images, _ = next(iter(train_loader))
images = images.to(device)
images = flatten(images)
reconstructed_images = decoder(encoder(images))
images = images.reshape(-1, 28, 28).cpu().detach().numpy()
reconstructed_images = reconstructed_images.reshape(-1, 28, 28).cpu().detach().numpy()

plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.axis('off')
    plt.subplot(2, 5, i+6)
    plt.imshow(reconstructed_images[i], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
