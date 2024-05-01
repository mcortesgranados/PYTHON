"""
01. Building and training deep neural networks for classification tasks.



"""

import tensorflow as tf
from tensorflow.keras import layers, models

# Define the architecture of the neural network
def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Load the dataset (example: Fashion MNIST)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape the data for the model
train_images = train_images.reshape((-1, 28*28))
test_images = test_images.reshape((-1, 28*28))

# Define model parameters
input_shape = train_images[0].shape
num_classes = len(set(train_labels))

# Create the model
model = create_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy:', test_acc)
