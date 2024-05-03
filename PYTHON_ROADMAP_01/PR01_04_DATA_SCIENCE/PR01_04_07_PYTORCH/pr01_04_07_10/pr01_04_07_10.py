"""
10. Training variational autoencoders (VAEs) for generative modeling and data synthesis.

Variational Autoencoders (VAEs) are generative models that learn to generate data by mapping it into a latent space and then 
reconstructing the data from samples drawn from this latent space. They consist of an encoder that maps input data into a 
distribution in latent space, and a decoder that reconstructs the data from samples drawn from this distribution. 
Let's implement a simple VAE in PyTorch for generating synthetic images based on the MNIST dataset.

In this example:

Imports: Import necessary libraries including PyTorch and torchvision.
Device Configuration: Set the device to GPU if available, else CPU.
Hyperparameters: Define hyperparameters such as input size, hidden size, latent size, etc.
Image Processing: Define transformations and load the MNIST dataset.
Variational Autoencoder (VAE): Define the VAE model with encoder and decoder components.
Loss Function: Define the VAE loss function, which consists of reconstruction loss (binary cross-entropy) and KL divergence.
Training Loop: Train the VAE model using the MNIST dataset.
Generate Synthetic Images: Generate synthetic images by sampling random points from the latent space and decoding them.
Visualization: Visualize the generated images using matplotlib.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784  # 28x28 images from MNIST dataset
hidden_size = 256
latent_size = 64
num_epochs = 20
batch_size = 100
learning_rate = 1e-3

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

# Define Variational Autoencoder (VAE) model
class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2_mean = nn.Linear(hidden_size, latent_size)
        self.fc2_logvar = nn.Linear(hidden_size, latent_size)
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        hidden = torch.relu(self.fc1(x))
        mean = self.fc2_mean(hidden)
        logvar = self.fc2_logvar(hidden)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        hidden = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(hidden))

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

# Loss function for VAE
def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Initialize VAE model
model = VAE(input_size, hidden_size, latent_size).to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0
    for i, (images, _) in enumerate(train_loader):
        images = images.view(-1, input_size).to(device)

        # Forward pass
        recon_images, mu, logvar = model(images)
        
        # Compute loss
        loss = vae_loss(recon_images, images, mu, logvar)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    # Print training statistics
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

# Generate synthetic images from random samples in latent space
with torch.no_grad():
    z = torch.randn(16, latent_size).to(device)
    generated_images = model.decode(z).view(-1, 1, 28, 28).cpu().numpy()

# Visualize generated images
plt.figure(figsize=(8, 2))
for i in range(16):
    plt.subplot(2, 8, i+1)
    plt.imshow(generated_images[i, 0], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
