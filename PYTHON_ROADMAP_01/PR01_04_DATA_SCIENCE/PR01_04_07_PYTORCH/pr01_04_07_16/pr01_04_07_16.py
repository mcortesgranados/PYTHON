"""
16. Implementing deep learning models for recommendation systems to provide personalized recommendations to users.

Implementing recommendation systems with deep learning models involves leveraging user-item interactions 
to generate personalized recommendations. One popular approach is to use Matrix Factorization techniques or 
deep learning architectures such as Neural Collaborative Filtering (NCF). 
Let's implement an example of a simple Neural Collaborative Filtering model using PyTorch.

In this example:

Sample Data: We define sample user-item interaction data consisting of user IDs, item IDs, and ratings.
Dataset Class: We define a custom RecommendationDataset class to load and process the interaction data.
NCF Model: We define a Neural Collaborative Filtering (NCF) model using PyTorch's nn.Module interface. The model consists of user and 
item embeddings followed by fully connected layers.
Training Loop: We train the NCF model using the interaction data and optimize it using stochastic gradient descent.
Prediction Function: We define a function to predict ratings for a given user and item pair using the trained model.
Example Usage: We demonstrate how to use the trained model to predict ratings for a specific user-item pair.

This example demonstrates how to implement a simple Neural Collaborative Filtering (NCF) model for recommendation systems using PyTorch. 
The model learns user and item embeddings and predicts ratings based on user-item interactions.


"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset

# Sample user-item interaction data (user IDs, item IDs, ratings)
user_ids = np.array([0, 0, 1, 1, 2, 3, 3, 4, 4, 4])
item_ids = np.array([0, 1, 1, 2, 2, 0, 2, 0, 1, 2])
ratings = np.array([5, 4, 3, 2, 5, 4, 2, 3, 4, 5])

# Define Dataset class
class RecommendationDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.user_ids[idx]), torch.tensor(self.item_ids[idx]), torch.tensor(self.ratings[idx])

# Define Neural Collaborative Filtering (NCF) model
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(2 * embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_ids, item_ids):
        user_embedded = self.user_embedding(user_ids)
        item_embedded = self.item_embedding(item_ids)
        concatenated = torch.cat([user_embedded, item_embedded], dim=1)
        output = self.fc_layers(concatenated)
        return output.squeeze(1)

# Initialize model, loss function, and optimizer
num_users = len(np.unique(user_ids))
num_items = len(np.unique(item_ids))
embedding_dim = 10
model = NCF(num_users, num_items, embedding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create DataLoader
dataset = RecommendationDataset(user_ids, item_ids, ratings)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for user_ids_batch, item_ids_batch, ratings_batch in loader:
        optimizer.zero_grad()
        outputs = model(user_ids_batch, item_ids_batch)
        loss = criterion(outputs, ratings_batch.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * user_ids_batch.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Function to predict ratings
def predict_rating(model, user_id, item_id):
    model.eval()
    with torch.no_grad():
        user_tensor = torch.tensor([user_id])
        item_tensor = torch.tensor([item_id])
        rating_prediction = model(user_tensor, item_tensor)
    return rating_prediction.item()

# Example usage
user_id = 0
item_id = 2
predicted_rating = predict_rating(model, user_id, item_id)
print(f'Predicted rating for user {user_id} and item {item_id}: {predicted_rating:.2f}')
