"""
15. Developing deep learning models for audio processing tasks like speech recognition or music generation.

Let's implement an example of training a deep learning model for speech recognition using PyTorch and the popular 
LibriSpeech dataset. We'll build a Convolutional Neural Network (CNN) with recurrent layers (RNNs or LSTMs) 
for processing audio spectrograms and performing speech recognition.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchaudio.datasets import LIBRISPEECH
from torchaudio.transforms import MelSpectrogram
from torchvision.transforms import Compose
import torchaudio

# Set device configuration (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations for preprocessing audio
transform = Compose([
    torchaudio.transforms.Resample(orig_freq=16_000, new_freq=8_000),
    MelSpectrogram(n_fft=400, win_length=400, hop_length=160, n_mels=64)
])

# Load LibriSpeech dataset
train_dataset = LIBRISPEECH('./', url='train-clean-100', download=True, transform=transform)

# Define DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Define Deep Learning Model for Speech Recognition (CNN-RNN)
class SpeechRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(SpeechRecognitionModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU()
        )
        self.rnn = nn.GRU(input_size=128, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x

# Initialize model, loss function, and optimizer
num_classes = len(train_dataset.classes)
model = SpeechRecognitionModel(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for spectrograms, targets in train_loader:
        spectrograms, targets = spectrograms.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(spectrograms)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * spectrograms.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'speech_recognition_model.pth')


