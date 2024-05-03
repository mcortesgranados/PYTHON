"""
27. Developing deep learning models for sequence-to-sequence tasks such as language translation or chatbot training.

Sequence-to-sequence (Seq2Seq) models are used for tasks where the input and output are both sequences of arbitrary lengths. 
They have applications in various tasks such as language translation, chatbot training, and text summarization. 
In this example, we'll implement a simple Seq2Seq model using an Encoder-Decoder architecture with Long Short-Term Memory (LSTM) 
cells in PyTorch for language translation.

In this example:

We define a simple sequence-to-sequence model (Seq2Seq) with LSTM cells for language translation.
We prepare training data consisting of input and target sequences.
We define the model, loss function (CrossEntropyLoss), and optimizer (Adam).
We train the model using the input and target sequences.
After training, we test the trained model by translating a given input sequence into the target language.

This example demonstrates how to implement a simple Seq2Seq model using an Encoder-Decoder architecture with LSTM cells in PyTorch 
for language translation tasks. In practice, more sophisticated Seq2Seq architectures like Attention Mechanisms are 
often used for better performance on complex sequence-to-sequence tasks.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a simple sequence-to-sequence model with LSTM cells
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(input_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, targets):
        # Encoder phase
        _, (encoder_hidden, encoder_cell) = self.encoder(inputs)

        # Decoder phase
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
        outputs = []
        for target in targets:
            output, (decoder_hidden, decoder_cell) = self.decoder(target.unsqueeze(0), (decoder_hidden, decoder_cell))
            output = self.fc(output.squeeze(0))
            outputs.append(output)
        return torch.stack(outputs, dim=1)

# Define a function to prepare sequences for training
def prepare_sequence(seq, vocab):
    idxs = [vocab.index(w) for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

# Define training data
input_seqs = ['hello', 'how', 'are', 'you']
target_seqs = ['bonjour', 'comment', 'vas', 'tu']
vocab = sorted(set(input_seqs + target_seqs))
input_size = len(vocab)
target_size = len(vocab)

# Convert input and target sequences into tensors
input_tensor = prepare_sequence(input_seqs, vocab)
target_tensor = prepare_sequence(target_seqs, vocab)

# Define model, loss function, and optimizer
hidden_size = 128
model = Seq2Seq(input_size, hidden_size, target_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(input_tensor.unsqueeze(1).float(), target_tensor.unsqueeze(1))
    loss = criterion(outputs.view(-1, target_size), target_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Test the trained model
with torch.no_grad():
    input_test = 'how'
    input_tensor_test = prepare_sequence(input_test, vocab)
    output_tensor_test = model(input_tensor_test.unsqueeze(0).float(), torch.zeros(1, 1, hidden_size))
    _, predicted_indexes = torch.max(output_tensor_test, dim=2)
    predicted_indexes = predicted_indexes.squeeze().numpy()
    predicted_sequence = [vocab[idx] for idx in predicted_indexes]
    print(f'Translated sequence for "{input_test}": {" ".join(predicted_sequence)}')
