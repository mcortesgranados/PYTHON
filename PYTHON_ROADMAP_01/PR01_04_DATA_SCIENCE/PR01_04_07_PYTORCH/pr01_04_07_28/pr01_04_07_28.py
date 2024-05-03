"""
28. Implementing deep learning models for emotion recognition tasks to classify emotions from facial expressions or text.

Emotion recognition tasks involve classifying emotions from inputs such as facial expressions or text. Deep learning models are commonly used for such tasks, 
particularly Convolutional Neural Networks (CNNs) for image-based emotion recognition and Recurrent Neural Networks (RNNs) for text-based emotion recognition. 
In this example, we'll implement a simple CNN-based model for image-based emotion recognition using PyTorch and the FER2013 dataset.

In this example:

We define a simple CNN architecture (EmotionCNN) for emotion recognition, with three convolutional layers followed by fully connected layers.
We load the FER2013 dataset, which contains facial expression images labeled with seven emotions.
We prepare data loaders for training and testing, which batch and shuffle the dataset.
We initialize the model, loss function (CrossEntropyLoss), and optimizer (Adam).
We train the model on the training set for a fixed number of epochs.
After training, we evaluate the model's accuracy on the test set to assess its performance.

This example demonstrates how to implement a simple CNN-based model for image-based emotion recognition using PyTorch and the FER2013 dataset. 
In practice, more sophisticated architectures and larger datasets may be used for real-world emotion recognition tasks. 
Additionally, text-based emotion recognition tasks can be approached similarly, with appropriate preprocessing of text data and the use of RNNs 
or Transformer models.


"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Define a simple convolutional neural network (CNN) for emotion recognition
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 128 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load FER2013 dataset for emotion recognition
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.FER2013(root='./data', train=True, download=True, transform=transform)
testset = datasets.FER2013(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = DataLoader(testset, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
num_classes = 7  # 7 emotions in FER2013 dataset
model = EmotionCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(trainset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}')

# Evaluate the model on the test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy on the test set: {accuracy:.2%}')
