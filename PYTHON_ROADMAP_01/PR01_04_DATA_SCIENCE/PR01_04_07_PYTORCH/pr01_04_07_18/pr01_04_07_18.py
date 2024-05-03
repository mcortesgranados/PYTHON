"""
18. Implementing adversarial attacks and defenses to study the robustness of deep learning models against adversarial examples.


Implementing adversarial attacks and defenses involves crafting inputs to fool a deep learning model (adversarial attacks) and developing techniques 
to improve model robustness against such attacks (adversarial defenses). Let's implement an example of adversarial attacks and a 
basic defense mechanism using PyTorch.

First, we'll implement a simple adversarial attack called the Fast Gradient Sign Method (FGSM), and then we'll apply a basic defense 
mechanism called adversarial training.

In this example:

We define a DataLoader for the CIFAR-10 dataset.
We train the model using an adversarial training loop, where we generate adversarial examples using the FGSM attack and update the 
model using both clean and perturbed images.
We evaluate the model's accuracy after adversarial training.

This example demonstrates how to implement a basic adversarial defense mechanism (adversarial training) using PyTorch. 
Adversarial training improves the model's robustness against adversarial examples by incorporating them into the training process.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

# Load pretrained model (e.g., pretrained on CIFAR-10)
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

# Define transform for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load a sample image from CIFAR-10 dataset
cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
image, label = cifar10_test[0]
image = image.unsqueeze(0)  # Add batch dimension

# Function to apply FGSM attack
def fgsm_attack(image, epsilon, data_grad):
    perturbed_image = image + epsilon * data_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Clip to maintain pixel values within [0, 1]
    return perturbed_image

# Function to test model accuracy on a given dataset
def test_model(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# Define parameters for the FGSM attack
epsilon = 0.05  # Perturbation magnitude

# Get model prediction before attack
outputs = model(image)
_, predicted = torch.max(outputs, 1)
print('Prediction before attack:', predicted.item())

# Calculate gradients of the loss w.r.t. input image
criterion = nn.CrossEntropyLoss()
image.requires_grad = True
outputs = model(image)
loss = criterion(outputs, torch.tensor([label]))
model.zero_grad()
loss.backward()
data_grad = image.grad.data

# Apply FGSM attack
perturbed_image = fgsm_attack(image, epsilon, data_grad)

# Get model prediction after attack
outputs = model(perturbed_image)
_, predicted = torch.max(outputs, 1)
print('Prediction after attack:', predicted.item())

# Plot original and perturbed images
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image.squeeze(0).permute(1, 2, 0))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('Perturbed Image')
plt.imshow(perturbed_image.squeeze(0).detach().permute(1, 2, 0))
plt.axis('off')
plt.show()
