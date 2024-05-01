"""
01. Building and training deep neural networks for classification tasks.



"""

import tensorflow as tf
from tensorflow import keras

# Sample data (replace with your data source)
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
x_train, y_train = mnist.data / 255.0, mnist.target.astype(np.int32)
x_test, y_test = mnist.data[70000:] / 255.0, mnist.target[70000:].astype(np.int32)

# Define the classification model using Keras Sequential API
model = keras.Sequential([
    # Input layer for 28x28 pixel MNIST images (flatten 2D data to 1D vector)
    keras.Input(shape=(28 * 28,)),

    # Hidden layers with activation functions
    keras.layers.Dense(units=32, activation='relu'),  # First hidden layer with 32 neurons and ReLU activation
    keras.layers.Dense(units=16, activation='relu'),  # Second hidden layer with 16 neurons and ReLU activation

    # Output layer with softmax activation for 10 digits (0-9)
    keras.layers.Dense(units=10, activation='softmax')
])

# Compile the model with appropriate loss function, optimizer, and metrics
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
pip install tensorflow

# Print model summary for better understanding of architecture
print(model.summary())

# Train the model on the training data (more epochs for better accuracy)
model.fit(x_train, keras.utils.to_categorical(y_train), epochs=15, batch_size=32, validation_data=(x_test, keras.utils.to_categorical(y_test)))

# Evaluate the model's performance on the test data
test_loss, test_acc = model.evaluate(x_test, keras.utils.to_categorical(y_test))
print('Test accuracy:', test_acc)

# Save the trained model for future use (optional)
model.save('my_mnist_classifier.h5')
