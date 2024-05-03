"""
30. Implementing deep learning models for transfer learning tasks in domains like computer vision, natural language processing, or audio processing.

Transfer learning is a technique widely used in deep learning, where a model trained on one task is reused as the starting point for a 
model on a second task. It's particularly useful when you have a small dataset for your target task but a large dataset for a related task. 
In this example, we'll demonstrate how to implement transfer learning using a pre-trained convolutional neural network (CNN) for a computer 
vision task, specifically image classification, using PyTorch and the CIFAR-10 dataset.

In this example:

We define transformations for the CIFAR-10 dataset to prepare the data for the pre-trained model.
We load the CIFAR-10 dataset for training and testing.
We load a pre-trained ResNet model from the torchvision model zoo.
We freeze the parameters of the pre-trained model so that they are not updated during training.
We modify the output layer of the model to match the number of classes in CIFAR-10.
We define a loss function (CrossEntropyLoss) and optimizer (SGD) for training the model.
We train the model on the CIFAR-10 training set for a fixed number of epochs.
We evaluate the trained model on the CIFAR-10 test set to measure its accuracy.

This example demonstrates how to implement transfer learning using a pre-trained CNN for image classification tasks in PyTorch. 
You can extend this approach to other domains like natural language processing or audio processing by using different pre-trained models and datasets
suited for those tasks.


"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to fit pre-trained model input size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = DataLoader(testset, batch_size=64, shuffle=False)

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Freeze the parameters of the pre-trained model
for param in model.parameters():
    param.requires_grad = False

# Modify the output layer to match the number of classes in CIFAR-10 (10 classes)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(trainset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}')

# Evaluate the model on the test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy on the test set: {accuracy:.2%}')
