"""
22. Implementing federated learning algorithms for training machine learning models on decentralized data sources.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import syft as sy

# Define a simple linear model
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

# Initialize workers (simulated decentralized clients)
hook = sy.TorchHook(torch)
alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")
workers = [alice, bob]

# Generate toy data and distribute to workers
X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])
data = [(X[i], y[i]) for i in range(len(X))]

# Distribute data to workers
data_ptr = [sy.BaseDataset([datapoint]).send(worker) for worker, datapoint in zip(workers, data)]

# Define a function for federated averaging
def federated_averaging(model, data_ptrs, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        for data_ptr in data_ptrs:
            # Train on each worker's data
            model.send(data_ptr.location)
            optimizer.zero_grad()
            inputs, labels = data_ptr[0], data_ptr[1]
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            model.get()

        # Aggregate model parameters using federated averaging
        with torch.no_grad():
            for param in model.parameters():
                param_avg = sum([worker_model.get().data for worker_model in model.buffers()]) / len(workers)
                param.set_(param_avg)

        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Initialize model, optimizer, and loss function
model = LinearModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Perform federated averaging
federated_averaging(model, data_ptr, optimizer, criterion, num_epochs=100)

# Evaluate model
with torch.no_grad():
    model.eval()
    for worker in workers:
        model.move(worker)
        inputs, labels = worker._objects[0][0], worker._objects[0][1]
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        print(f"Worker {worker.id}: Loss - {loss.item()}, Predictions - {outputs}")
