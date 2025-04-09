# For AI 681 students only
# Convolutional Neural Networks (CNN)
# Multi-class classification for recognizing handwritten digits
#  

# Import some libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential, layers, Input
import matplotlib.pyplot as plt

# Load training and testing data from the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
 
# Choose an image for plotting 
index = 0  # Change this to display a different image

# Print the pixel values of the image 
print(train_images[index].reshape(28, 28))

# Plot the chosen image
plt.imshow(train_images[index].reshape(28, 28), cmap="gray")  # Already 2D (28, 28) from the dataset
plt.title(f"Label: {train_labels[index]}")  # Use non-encoded label directly
plt.axis("off")  # Hide axes for better visualization
plt.show()

# Preprocessing the data
# Normalizes pixel values by dividing by 255 so that all values are in the range [0, 1]
train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255

print("Plotting first label before encoding")
print(train_labels[0])

# One-hot encode the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

print("Plotting first label after encoding")
print(train_labels[0])

# Define convolutinal layers of the CNN model
model = Sequential()

# Add an input layer  
model.add(Input(shape=(28, 28, 1)))  

# add a convolutional layer with 32 filters, each of size (3, 3).
model.add(layers.Conv2D(32, (3, 3), activation="relu"))   

# add a MaxPooling layer 
model.add(layers.MaxPooling2D((2, 2)))                                             
model.add(layers.Conv2D(64, (3, 3), activation="relu"))                            
model.add(layers.MaxPooling2D((2, 2)))                                             
model.add(layers.Conv2D(64, (3, 3), activation="relu"))

# Converts the multi-dimensional output of the previous layer into a 1D vector 
model.add(layers.Flatten())

# Add a fully-connected hidden (dense) layer 
model.add(layers.Dense(64, activation="relu"))

# Add an output layer 
model.add(layers.Dense(10, activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")
 
# Get the weights of the first Conv2D layer
filters, biases = model.layers[0].get_weights()

# Print filter shape
print("Filter shape:", filters.shape)  # Should be (3, 3, 1, 32) for (height, width, channels, num_filters)

# Print first filter values
print("First filter:\n", filters[:, :, 0, 0])  # First filter applied to channel 0


