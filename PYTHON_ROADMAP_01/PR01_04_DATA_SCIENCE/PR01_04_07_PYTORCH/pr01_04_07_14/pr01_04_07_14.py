"""
14. Training deep learning models for image segmentation tasks such as medical image analysis or semantic segmentation.

Image segmentation is a computer vision task that involves partitioning an image into multiple segments or regions to 
simplify the representation of an image. This is useful for tasks such as object detection, medical image analysis, 
and semantic segmentation. Let's implement an example of training a deep learning model for semantic segmentation 
using PyTorch and the popular Cityscapes dataset.

In this example:

Imports: Import necessary libraries including PyTorch, torchvision, and matplotlib.
Device Configuration: Set the device to GPU if available, else CPU.
Define Data Transformations: Define transformations for data augmentation and normalization.
Load Cityscapes Dataset: Load the Cityscapes dataset for semantic segmentation.
Define DataLoader: Create a DataLoader to load the dataset in batches.
Define Deep Learning Model for Semantic Segmentation: Define a Fully Convolutional Network (FCN) model for semantic segmentation.
Initialize Model, Loss Function, and Optimizer: Initialize the FCN model, Cross Entropy loss function, and Adam optimizer.
Train the Model: Train the FCN model using the Cityscapes dataset.
Function to Visualize Predictions: Define a function to visualize predictions made by the trained model.
Visualize Predictions: Visualize predictions made by the trained model on a sample of images from the dataset.

This example demonstrates how to implement and train a deep learning model for semantic segmentation using PyTorch on the Cityscapes dataset. 
The model is trained to predict pixel-wise segmentation masks for images, and the predictions are visualized for inspection.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set device configuration (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations for data augmentation and normalization
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Cityscapes dataset
train_dataset = datasets.Cityscapes(root='./data', split='train', mode='fine', target_type='semantic', transform=data_transforms)

# Define DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Define Deep Learning Model for Semantic Segmentation (e.g., FCN)
class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        self.features = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(self.features.children())[:-2])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.decoder(x)
        return x

# Initialize model, loss function, and optimizer
num_classes = len(train_dataset.classes)
model = FCN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Function to visualize predictions
def visualize_predictions(model, loader, num_images=5):
    model.eval()
    for i, (images, labels) in enumerate(loader):
        if i == num_images:
            break
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        for j in range(images.size(0)):
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title('Original Image')
            plt.imshow(images[j].permute(1, 2, 0).cpu())
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.title('Predicted Segmentation')
            plt.imshow(preds[j].cpu(), cmap='tab20', interpolation='none')
            plt.axis('off')
            plt.show()

# Visualize predictions
visualize_predictions(model, train_loader)
