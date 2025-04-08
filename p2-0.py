# For AI 681 students only 
# Project 2 Starter Code 
 
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
 
# Derivatives of activation functions
 

# Initialize neural network parameters
 

# Initialization of weights 
 

# Function to compute total loss on the entire dataset
 

# Training  
for epoch in range(epochs):
   
    
 
print("\nTraining Complete!\n")

# ---- Model Evaluation ----
 

# Predictions
 

# Compute accuracy
 

# Plot cost function convergence history 
 

