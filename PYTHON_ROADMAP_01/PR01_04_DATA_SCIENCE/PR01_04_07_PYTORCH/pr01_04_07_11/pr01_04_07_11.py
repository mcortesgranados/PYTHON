"""
11. Implementing graph neural networks (GNNs) for graph-structured data analysis.



"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

# Device configuration (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 200
learning_rate = 0.01
hidden_size = 16

# Load Cora dataset
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]  # Get the first graph in the dataset

# Define Graph Convolutional Network (GCN) model
class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, output_size)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize GCN model
model = GCN(input_size=data.num_features, hidden_size=hidden_size, output_size=dataset.num_classes).to(device)

# Loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Data loader
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = criterion(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    avg_loss = total_loss / len(dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
for data in loader:
    data = data.to(device)
    output = model(data.x, data.edge_index)
    _, predicted = torch.max(output[data.test_mask], 1)
    total += data.y[data.test_mask].size(0)
    correct += (predicted == data.y[data.test_mask]).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")
