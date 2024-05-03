"""
25. Training deep learning models for face recognition tasks to identify individuals from images or videos.

Face recognition tasks involve identifying individuals from images or videos. Deep learning models such as Convolutional Neural Networks (CNNs) 
are commonly used for face recognition. In this example, we'll train a CNN-based face recognition model using PyTorch and the CIFAR-10 dataset 
as a simplified example.

In this example:

We define a simple CNN architecture (SimpleCNN) for face recognition. This architecture consists of several convolutional layers followed 
by fully connected layers.

We load the CIFAR-10 dataset, which contains images of various objects, and apply transformations to prepare the data for training.

We initialize the model, loss function (CrossEntropyLoss), and optimizer (Adam).

We train the model on the CIFAR-10 training set for a fixed number of epochs.

After training, we evaluate the model's accuracy on the test set to assess its performance.

While CIFAR-10 is not a dataset specifically for face recognition, this example serves as a simplified illustration of training a 
CNN-based model for classification tasks like face recognition. In practice, face recognition tasks typically involve larger datasets 
containing images of faces, and more sophisticated CNN architectures are employed.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets

# Define a simple convolutional neural network (CNN) for face recognition
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 10)  # Assuming 10 classes for simplicity (not individual faces)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load CIFAR-10 dataset and apply transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
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
    epoch_loss = running_loss / len(train_loader.dataset)
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
