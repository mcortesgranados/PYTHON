"""
07. Fine-tuning pre-trained models for transfer learning tasks on new datasets.

 Transfer learning involves taking a pre-trained model and adapting it to a new, possibly related, task. 
 This is especially useful when you have a small dataset for the new task, as the pre-trained model has already 
 learned useful features from a large dataset. Here's an example of fine-tuning a pre-trained ResNet model on the CIFAR-10 dataset using PyTorch.

 In this example:

Imports: Import necessary libraries including PyTorch and torchvision.
Device Configuration: Set the device to GPU if available, else CPU.
Hyperparameters: Define hyperparameters such as number of classes, number of epochs, etc.
Data Processing: Define transformations and load CIFAR-10 dataset using torchvision.
Pre-trained Model: Load a pre-trained ResNet model from torchvision.models.
Modify Model: Replace the final fully connected layer to match the number of classes in CIFAR-10 and freeze other layers.
Loss Function and Optimizer: Define CrossEntropyLoss and Adam optimizer.
Training Loop: Loop through the dataset for multiple epochs, compute loss, and update model parameters.
Evaluation Loop: Evaluate the model on the test set and compute accuracy.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models

# Device configuration (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_classes = 10
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Image processing and augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match input size of ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet mean and std
])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Load pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Freeze all layers except the final fully connected layer
for param in model.parameters():
    param.requires_grad = False

# Modify the final fully connected layer to match the number of classes in CIFAR-10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move model to device
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to calculate accuracy
def accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute training statistics
        running_loss += loss.item() * images.size(0)
        running_accuracy += accuracy(outputs, labels) * images.size(0)
    
    # Print training statistics
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = running_accuracy / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

# Evaluation loop
model.eval()
test_accuracy = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Compute test accuracy
        test_accuracy += accuracy(outputs, labels) * images.size(0)

# Print test accuracy
test_accuracy /= len(test_loader.dataset)
print(f"Test Accuracy: {test_accuracy:.4f}")
