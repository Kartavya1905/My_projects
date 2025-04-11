import numpy as np
import tensorflow as tf
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset from UCI repository (can also use a local CSV file)
datafile = "data_banknote_authentication.txt"
column_names = ['variance', 'skewness', 'kurtosis', 'entropy', 'class']
data = pd.read_csv(datafile, header=None, names=column_names)

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Features and target
X = data.drop('class', axis=1).values
y = data['class'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)), 
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # sigmoid for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model without shuffling
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2,shuffle=False, verbose=1)

# Evaluate on test data
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {accuracy:.4f}")

# Plot training history
plt.plot(history.history['accuracy'], label='Training Acc')
plt.plot(history.history['val_accuracy'], label='Validation Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Banknote Authentication - Training Accuracy')
plt.show()
