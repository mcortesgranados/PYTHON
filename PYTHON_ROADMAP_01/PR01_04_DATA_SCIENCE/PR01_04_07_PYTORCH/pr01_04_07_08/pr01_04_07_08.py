"""
08. Implementing siamese networks for tasks like face recognition or similarity learning.

In this example:

Imports: Import necessary libraries including PyTorch, torchvision, and other utilities.

Device Configuration: Set the device to GPU if available, else CPU.

Hyperparameters: Define hyperparameters such as number of epochs, learning rate, etc.

Siamese Network: Define the Siamese network architecture consisting of convolutional layers followed by fully connected layers.

Contrastive Loss: Define the contrastive loss function used to train the Siamese network.

Custom Dataset: Create a custom dataset class for Olivetti Faces dataset to generate pairs of images with their labels.

Load Dataset: Load the Olivetti Faces dataset and split it into train and test sets.

Train Function: Define the training function to train the Siamese network.

Main Function: Load the dataset, define model, loss function, and optimizer, and train the Siamese network. 
Then evaluate the model on the test set and print the accuracy.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Device configuration (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 20
batch_size = 32
learning_rate = 0.001
embedding_size = 64

# Define Siamese network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*6*6, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, embedding_size)
        )

    def forward_one(self, x):
        return self.cnn(x)

    def forward(self, x1, x2):
        output1 = self.forward_one(x1)
        output2 = self.forward_one(x2)
        return output1, output2

# Contrastive loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                       (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# Custom dataset for Olivetti Faces
class OlivettiFacesDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1, label1 = self.data[idx]
        img2, label2 = self.get_positive_sample(label1, idx)
        label = torch.tensor(1 if label1 == label2 else 0, dtype=torch.float32)
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

    def get_positive_sample(self, label, idx):
        # Get a random positive sample with the same label
        pos_idx = idx
        while pos_idx == idx or self.data[pos_idx][1] != label:
            pos_idx = np.random.randint(len(self.data))
        return self.data[pos_idx]

# Load Olivetti Faces dataset
def load_olivetti_faces():
    data = torchvision.datasets.OlivettiFaces(root='./data', download=True)
    return data

# Train function
def train(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for images1, images2, labels in train_loader:
        images1 = images1.to(device)
        images2 = images2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output1, output2 = model(images1, images2)
        loss = criterion(output1, output2, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

# Main function
def main():
    # Load dataset
    data = load_olivetti_faces()
    
    # Split dataset into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize images
    ])
    
    # Create custom datasets
    train_dataset = OlivettiFacesDataset(train_data, transform=transform)
    test_dataset = OlivettiFacesDataset(test_data, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")
    
    # Evaluation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images1, images2, labels in test_loader:
            images1 = images1.to(device)
            images2 = images2.to(device)
            labels = labels.to(device)
            
            output1, output2 = model(images1, images2)
            euclidean_distance = nn.functional.pairwise_distance(output1, output2)
            predictions = (euclidean_distance < 0.5).float()
            
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
