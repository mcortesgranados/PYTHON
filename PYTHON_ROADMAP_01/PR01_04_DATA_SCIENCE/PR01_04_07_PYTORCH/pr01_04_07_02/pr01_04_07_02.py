"""
02. Implementing convolutional neural networks (CNNs) for image classification and object detection.

Implementing convolutional neural networks (CNNs) for image classification and object detection using 
PyTorch involves defining a CNN architecture, loading and preprocessing the dataset, defining the loss function and optimizer, 
and then iterating through the dataset to train the model. In this example, we'll build a simple CNN for classifying images from the CIFAR-10 dataset.

Explanation:

We import the necessary libraries, including PyTorch and torchvision for handling datasets and transformations.

We define the convolutional neural network architecture as a class CNN that inherits from nn.Module. 
This simple CNN consists of three convolutional layers with ReLU activation functions followed by max-pooling layers, and two fully connected layers.

We load the CIFAR-10 dataset using torchvision, apply transformations such as converting images to tensors and normalizing the pixel values.

We initialize the CNN, define the loss function (CrossEntropyLoss), and choose an optimizer (SGD with momentum) to train the network.

We iterate through the dataset for a specified number of epochs, loading batches of data, computing the predictions of the CNN, 
calculating the loss, performing backpropagation, and updating the network parameters using the optimizer.

During training, we print the average loss every 1000 mini-batches.

After training, we print "Finished Training" to indicate that the training process is complete.

This example demonstrates how to build and train a simple CNN for image classification tasks using PyTorch. 
With PyTorch, you have flexibility in defining and customizing CNN architectures and training procedures 
to suit your specific image classification or object detection tasks.

"""

# Importing necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Define the convolutional neural network architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load and preprocess the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

# Initialize the CNN, loss function, and optimizer
net = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training the CNN
for epoch in range(5):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:    # print every 1000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

print('Finished Training')
