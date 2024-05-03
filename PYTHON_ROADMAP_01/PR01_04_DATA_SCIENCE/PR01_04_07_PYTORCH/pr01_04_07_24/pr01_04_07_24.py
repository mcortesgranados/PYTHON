"""
24. Implementing neural architecture search (NAS) algorithms for automatically discovering optimal neural network architectures.

Neural Architecture Search (NAS) is a technique that automates the process of discovering optimal neural network architectures. 
There are various NAS algorithms, ranging from evolutionary algorithms to reinforcement learning-based approaches. 
In this example, we'll implement a simple form of NAS called Random Search for discovering optimal convolutional neural network (CNN) 
architectures for image classification using PyTorch.

In this example:

We define a simple CNN architecture (SimpleCNN) for image classification on the CIFAR-10 dataset.
We define a function (evaluate_model) to evaluate the accuracy of a given model on the test set.
We randomly sample hyperparameters for the CNN architecture and perform random search to find the best model based on accuracy.
For each trial, we train a randomly generated CNN architecture for a fixed number of epochs and evaluate its accuracy on the test set.
We keep track of the best model based on the highest accuracy achieved during the random search.
Finally, we save the best model to a file.

This example demonstrates how to perform neural architecture search using a simple random search approach for finding optimal 
CNN architectures for image classification tasks using PyTorch.


"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import random

# Define a simple convolutional neural network (CNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define function to evaluate model accuracy
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Set device (GPU if available, else CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define transformations and load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Perform random search for neural architecture
best_accuracy = 0.0
best_model = None
for _ in range(10):  # Perform 10 trials
    # Randomly sample hyperparameters
    num_conv_layers = random.randint(1, 3)
    num_filters = [random.randint(16, 64) for _ in range(num_conv_layers)]

    # Build CNN architecture based on sampled hyperparameters
    class RandomCNN(nn.Module):
        def __init__(self, num_conv_layers, num_filters):
            super(RandomCNN, self).__init__()
            self.conv_layers = nn.ModuleList()
            in_channels = 3
            for out_channels in num_filters:
                self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                self.conv_layers.append(nn.ReLU())
                self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                in_channels = out_channels
            self.fc = nn.Linear(in_channels * 4 * 4, 10)  # Assuming input size is 32x32

        def forward(self, x):
            for layer in self.conv_layers:
                x = layer(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    # Initialize model, loss function, and optimizer
    model = RandomCNN(num_conv_layers, num_filters).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    for epoch in range(5):  # Train for 5 epochs
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/5], Loss: {epoch_loss:.6f}')

    # Evaluate model accuracy
    accuracy = evaluate_model(model, test_loader, device)
    print(f'Accuracy: {accuracy}')

    # Update best model if accuracy improves
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print(f'Best Accuracy: {best_accuracy}')

# Save the best model
torch.save(best_model.state_dict(), 'best_model.pth')
