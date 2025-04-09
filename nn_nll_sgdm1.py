# For AI 681 students only 
# Feed-forward neural network with back propagation  
# Classification problem  

import numpy as np
import matplotlib.pyplot as plt

# Generate a simple dataset
np.random.seed(0)
N = 100  # Number of points per class
D = 2    # Dimension of input data point
K = 2    # Number of classes

# Create the dataset for training and testing
X = np.zeros((N*K, D))  # Input data
y = np.zeros(N*K, dtype='uint8')  # Labels

for j in range(K):
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0, 1, N)  # Radius
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2  # Theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)] # transfer to vertical matrix 
    y[ix] = j

# Plotting the data set
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()


# Neural network parameters
h = 5                              # Number of units in the hidden layer 
W1 = 0.01 * np.random.randn(D, h)  # Weight matrix for input to hidden layer
b1 = np.zeros((1, h))              # Bias for hidden layer
W2 = 0.01 * np.random.randn(h, K)  # Weight matrix for hidden to output layer
b2 = np.zeros((1, K))              # Bias for output layer

# Hyperparameters
learning_rate = 0.015
num_iterations = 100000
batch_size = 1   
 
def softmax(x):
    exp_x = np.exp(x - np.max(x))   
    return exp_x / np.sum(exp_x)

def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-12))  # Stability trick  
  
# Function to compute total loss on the entire dataset
def compute_total_loss(W1, b1, W2, b2):
    total_loss = 0
    n = X.shape[0]

    for i in range(n):
        Xi = X[i :i+1]  # Single sample (1, features)
        yi = y[i]  # Single target 
        
        # convert yi to one hot vector 
        yihot = np.zeros((1, 2))        
        yihot[0, yi] = 1     
        
        # Forward pass
        Z1 = np.dot(Xi, W1) + b1
        A1 = np.maximum(0, Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = softmax(Z2)
        
        total_loss += cross_entropy(yihot, A2)
 

    return total_loss / n  # Return average loss

cost_history = []  # Store cost for monitoring
    
# Training the neural network with SGD
for i in range(num_iterations):
    # Sample one data point
    randi = np.random.choice(N*K, batch_size)  # Choose N = batch_size  
    X_batch = X[randi]   # (N x D)
    y_batch = y[randi]   # (N x 1)

    if i == 1: 
        print('printing batch data at i=1:')        
        print(X_batch)
        print(y_batch)
    
    # ---- Forward pass ----
    # Layer 1 (Input -> Hidden)
    Z1 = np.dot(X_batch, W1) + b1  # Pre-activation (N x h)
    A1 = np.maximum(0, Z1)         # Activation (ReLU) (N x h)

    # Layer 2 (Hidden -> Output)
    Z2 = np.dot(A1, W2) + b2       # Pre-activation (N x K)

    # Calculate Negative Log-Likelihood Loss by using softmax 
    Z2 -= np.max(Z2, axis=1, keepdims=True)  # For numerical stability,  (N x K) 
    exp_scores = np.exp(Z2)                  # (N x K)  
    log_probs = Z2 - np.log(np.sum(exp_scores, axis=1, keepdims=True))  # \ln(a^2) (N x K)
    
    # Print loss every 1000 iterations
    if i % 2000 == 0:
        # Compute total loss 
        total_loss = compute_total_loss(W1, b1, W2, b2)           
        cost_history.append(total_loss)        
        print(f"Iteration {i}: Average Loss = {total_loss:.4f}")
        
    # ---- Backward pass ----
    # Gradient of the loss with respect to Z2 (using log-softmax trick)
    dLdZ2 = np.exp(log_probs)                   # activatin vector a^(2), (N x K) 
    dLdZ2[range(batch_size), y_batch] -= 1      # d(L)/d(z^(2)),  (N x K)  
    dLdZ2 /= batch_size                         # (N x K)

    # Gradients for W2 and b2
    dLdW2 = np.dot(A1.T, dLdZ2)                   # (h, K) 
    dLdb2 = np.sum(dLdZ2, axis=0, keepdims=True)  # (1, K), no change due to N = 1

    # Gradient for hidden layer (backprop through ReLU)
    dLdA1 = np.dot(dLdZ2, W2.T)
    dLdZ1 = dLdA1 * (Z1 > 0)  # Backprop through ReLU
    
    if i == 1:
        print(dLdZ1.shape)
        
    # Gradients for W1 and b1
    dLdW1 = np.dot(X_batch.T, dLdZ1) 
    dLdb1 = np.sum(dLdZ1, axis=0, keepdims=True)
    
    if i == 1:
        print(dLdW1.shape)
        
    # ---- SGD parameter update ----
    W1 -= learning_rate * dLdW1
    b1 -= learning_rate * dLdb1
    W2 -= learning_rate * dLdW2
    b2 -= learning_rate * dLdb2

# Evaluate the trained model
Z1 = np.dot(X, W1) + b1                      # Z1                   
hidden_layer = np.maximum(0, Z1)             # A1
scores = np.dot(hidden_layer, W2) + b2       # Z2 
predicted_class = np.argmax(scores, axis=1)  # index of the max element
                                             # (no need to compute A2)     
print(f'Training accuracy: {np.mean(predicted_class == y)}')

# Plot the decision boundary
def plot_decision_boundary(pred_func):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()

# Prediction function
def predict(X):
    hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
    scores = np.dot(hidden_layer, W2) + b2
    return np.argmax(scores, axis=1)

plot_decision_boundary( predict )

# Plot cost function convergence history 
plt.plot(range(0, len(cost_history) * 2000, 2000), cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.show()