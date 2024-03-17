import tensorflow as tf
from tensorflow.keras import layers, models

# Define the model architecture
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),   # Input layer (flatten the input image)
    layers.Dense(128, activation='relu'),   # Hidden layer with 128 neurons and ReLU activation function
    layers.Dense(10, activation='softmax')  # Output layer with 10 neurons for classification (softmax activation)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()
