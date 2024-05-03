"""
04. Developing generative adversarial networks (GANs) for generating synthetic data, images, or text.

This example demonstrates the basic structure of a GAN using PyTorch. Here's a brief overview of what each part does:

Imports: Import necessary libraries including PyTorch and torchvision.

Device Configuration: Set the device to GPU if available, else CPU.

Hyperparameters: Define hyperparameters such as latent size, hidden size, etc.

Data Loading: Load the MNIST dataset and create a data loader.

Network Initialization: Define the discriminator (D) and generator (G) neural networks.

Move Models to Device: Move the networks to the selected device (CPU or GPU).

Loss Function and Optimizers: Define binary cross-entropy loss and Adam optimizers for both networks.

Training Functions: Define functions to train the discriminator and generator.

Training Loop: Loop through the dataset, alternating between training the discriminator and generator.

Save Generated Images: Save generated images during training.

Save Model Checkpoints: Save the trained discriminator and generator models.

This code trains a GAN on the MNIST dataset to generate synthetic handwritten digit images. The discriminator learns to distinguish between 
real and fake images, while the generator learns to generate images that fool the discriminator.



"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device configuration (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
latent_size = 64
hidden_size = 256
image_size = 784  # 28x28 images from MNIST dataset
num_epochs = 200
batch_size = 100
sample_dir = 'samples'

# Create a directory to save generated images
import os
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

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

# Discriminator network
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())

# Generator network
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())

# Move models to device
D = D.to(device)
G = G.to(device)

# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

# Function to denormalize image
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# Function to train the discriminator
def train_discriminator(images):
    # Create real and fake labels
    real_labels = torch.ones(images.size(0), 1).to(device)
    fake_labels = torch.zeros(images.size(0), 1).to(device)
    
    # Train with real images
    outputs = D(images)
    d_loss_real = criterion(outputs, real_labels)
    real_score = outputs
    
    # Train with fake images
    z = torch.randn(images.size(0), latent_size).to(device)
    fake_images = G(z)
    outputs = D(fake_images)
    d_loss_fake = criterion(outputs, fake_labels)
    fake_score = outputs
    
    # Backpropagation and optimization
    d_loss = d_loss_real + d_loss_fake
    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()
    return d_loss, real_score, fake_score

# Function to train the generator
def train_generator():
    # Generate fake images and calculate loss
    z = torch.randn(batch_size, latent_size).to(device)
    fake_images = G(z)
    labels = torch.ones(batch_size, 1).to(device)
    g_loss = criterion(D(fake_images), labels)
    
    # Backpropagation and optimization
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()
    return g_loss, fake_images

# Start training
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        images = images.reshape(batch_size, -1).to(device)
        
        # Train discriminator
        d_loss, real_score, fake_score = train_discriminator(images)
        
        # Train generator
        g_loss, fake_images = train_generator()
        
        # Print some information
        if (i+1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch+1, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()))
    
    # Save generated images
    if (epoch+1) == 1:
        images = images.reshape(images.size(0), 1, 28, 28)
        torchvision.utils.save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))
        
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    torchvision.utils.save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

# Save the model checkpoints
torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')
