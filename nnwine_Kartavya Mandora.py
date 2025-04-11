# Project 2 Solution
# Name: Kartavya Mandora
 
# Import lib  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, delimiter=";")

# Select only 7 features and limit to 300 samples
selected_features = ["fixed acidity", "volatile acidity", "total sulfur dioxide", "residual sugar", "sulphates", "pH", "alcohol"]
df = df[selected_features + ["quality"]].sample(n=300, random_state=42)

# Convert quality scores to class labels (shift to start from 0)
y = df["quality"].values - 3  # Wine quality is from 3 to 8, shift to 0-5
X = df[selected_features].values
 
# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode labels
num_classes = len(np.unique(y))
#print(num_classes)

y_one_hot = np.zeros((len(y), num_classes))

for i, label in enumerate(y):
    y_one_hot[i, label] = 1

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Activation functions
def tanh(z):
    return np.tanh(z)
# Derivatives of activation functions
def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
# Implementing Cross-Entropy Loss function
def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]
# Initialize neural network parameters
input_size = X_train.shape[1]      # 7 features
hidden_size1 = 16
hidden_size2 = 12
output_size = num_classes          # 6 classes
iterations = 500
learning_rate = 0.01

# Initialization of weights 
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size1) * 0.01
b1 = np.zeros((1, hidden_size1))
W2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
b2 = np.zeros((1, hidden_size2))
W3 = np.random.randn(hidden_size2, output_size) * 0.01
b3 = np.zeros((1, output_size))

# Function to compute total loss on the entire dataset
loss_history = []

# Training  
for i in range(iterations):
    Z1 = np.dot(X_train, W1) + b1
    A1 = tanh(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = tanh(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = softmax(Z3)

    # Loss computation
    loss = cross_entropy(y_train, A3)
    loss_history.append(loss)

    # Backpropagation
    dZ3 = A3 - y_train
    dW3 = np.dot(A2.T, dZ3)
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * tanh_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * tanh_derivative(Z1)
    dW1 = np.dot(X_train.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Update weights and biases
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if (i+1) % 50 == 0:
        print(f"Epoch {i+1}/{iterations}, Loss: {loss:.4f}")
   
    
 
print("\nTraining Complete!\n")

# ---- Model Evaluation ----
Z1 = np.dot(X_test, W1) + b1
A1 = tanh(Z1)
Z2 = np.dot(A1, W2) + b2
A2 = tanh(Z2)
Z3 = np.dot(A2, W3) + b3
A3 = softmax(Z3)

# Predictions
y_pred = np.argmax(A3, axis=1)
y_true = np.argmax(y_test, axis=1)

# Compute accuracy
accuracy = np.mean(y_pred == y_true)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Plot the cost function convergence history 
plt.plot(loss_history)
plt.title("Loss over iterations")
plt.xlabel("i")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True)
plt.show() 

