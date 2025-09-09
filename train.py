import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load the Fashion MNIST dataset
# This dataset contains 70,000 grayscale images of 10 fashion categories.
# The images are 28x28 pixels.
print("Loading Fashion MNIST dataset...")
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Define the class names to use for a more readable output
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Preprocess the data
# Normalize the pixel values to be between 0 and 1.
# This is a common and important step for training neural networks.
print("Normalizing image data...")
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the Keras model
# We'll use a simple Sequential model with three layers:
# 1. Flatten layer to transform the 28x28 image to a 784-pixel array.
# 2. Dense hidden layer with 128 neurons and a ReLU activation function.
# 3. Dense output layer with 10 neurons, one for each class.
#    The softmax activation function ensures the output is a probability distribution.
print("Building and compiling the model...")
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
# We use the Adam optimizer and SparseCategoricalCrossentropy as the loss function.
# The 'accuracy' metric is used to evaluate the model's performance.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
# The model is trained on the training data for 10 epochs.
print("Training the model...")
model.fit(train_images, train_labels, epochs=10)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest accuracy: {test_acc}")

# Save the entire model to a single HDF5 file
# This file will be loaded by the Streamlit application.
model_filename = 'fashion_mnist_model.h5'
print(f"Saving the model to '{model_filename}'...")
model.save(model_filename)
print("Model saved successfully!")
