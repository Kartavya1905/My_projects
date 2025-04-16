# AI 680
# Project 2 Solution
# Name: Kartavya Mandora

# Import some libraries
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
 

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
# Load the dataset  
datafile = r"C:\Users\Lenovo\Downloads\Gitdemo\My_projects\students.csv"
df = pd.read_csv(datafile)

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Group final grades (G3) into 3 classes
# 0 = Low (0–9), 1 = Medium (10–14), 2 = High (15–20)
def grade_category(g3):
    if g3 <= 9:
        return 0  # Low
    elif g3 <= 14:
        return 1  # Medium
    else:
        return 2  # High

df['performance'] = df['G3'].apply(grade_category)
df = df.drop(columns=['G3'])  # Drop original grade

# Prepare features and labels
X = df.drop(columns=['performance'])
y = df['performance']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode the labels
y_encoded = to_categorical(y, num_classes=3)

# Split dataset (X_scaled, y_encoded) into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, shuffle=False) 

# Build a neural network with at least one hidden layer 
model = Sequential()
model.add(Dense(64, input_shape=(X.shape[1],), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 output classes

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, shuffle=False, verbose=1) 

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {accuracy:.2f}")

# Plot training history
plt.plot(history.history['accuracy'], label='Training Acc')
plt.plot(history.history['val_accuracy'], label='Validation Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Student Academic Performance - Training Accuracy')
plt.show()