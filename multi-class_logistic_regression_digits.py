# For AI 681 students only 
# Multi-class logistic regression for recognizing handwritten digits

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
from scipy.optimize import minimize

# Step 1: Load the digits dataset
digits = load_digits()

# Step 2: Visualize some of the images from the dataset
fig, axes = plt.subplots(1, 10, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap='gray')
    ax.set_title(f'{label}')
plt.show()

# Step 3: Prepare the data
# Flatten the images, converting each 8x8 image into a 64-element vector
X = digits.data
y = digits.target

print("Printing X and y before stacking")
print(X)
print(y)

# One-hot encode the target labels for multi-class classification
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Add a column of ones to X to account for the intercept term
X = np.hstack([np.ones((X.shape[0], 1)), X])

print("Printing X and y after stacking and encoding")
print(X)
print(y_onehot )

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Step 4: Define the negative log-likelihood loss function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # For numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def negative_log_likelihood_loss(W, X, y):
    n, d = X.shape
    K = y.shape[1]
    
    W = W.reshape((d, K))
    z = np.dot(X, W)      # n, K     
    g = softmax(z)
    
    # Calculate the negative log-likelihood
    loss = -np.sum(y * np.log(g + 1e-9)) / n  # Add 1e-9 to avoid log(0)
    return loss

# Step 5: Initialize weights and optimize using gradient descent
n_features = X_train.shape[1]
n_classes = y_train.shape[1]
initial_W = np.zeros(n_features * n_classes)

# Use the scipy.optimize.minimize function to minimize the negative log-likelihood loss
result = minimize(negative_log_likelihood_loss, initial_W, args=(X_train, y_train), method='L-BFGS-B', options={'maxiter': 2000})
optimized_W = result.x.reshape((n_features, n_classes))

# Step 6: Make predictions on the test set
z_test = np.dot(X_test, optimized_W)
g_test = softmax(z_test)
y_pred = np.argmax(g_test, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

print("Printing y_pred:")
print(y_pred)
print("Printing y_test_labels:")
print(y_test_labels)

# Step 7: Evaluate the model's performance
print("Accuracy:", accuracy_score(y_test_labels, y_pred))

# Step 8: Confusion Matrix visualization
conf_matrix = confusion_matrix(y_test_labels, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Step 9: Print out the learned coefficients including the intercept term
# print("Learned Coefficients (Weights including intercept):")
# print(optimized_W)
