# For AI 681 students only 
# Feed-forward neural network with back propagation  
# regression problem with PyTorch
# 
# To use PyTorch, you must install it with pip like this, 
# pip3 install torch torchvision torchaudio
# 
# and currently, PyTorch on Windows only supports Python 3.9-3.12
# For more info, please check https://pytorch.org/get-started/locally/


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
 
torch.manual_seed(0)

# Create a sample dataset with n data points
n = 50

# inputs 
x_input = torch.rand(n, 1)*torch.pi*2

# targets
y_output = torch.sin(x_input)  + 0.25 * torch.rand(n, 1)

# Plotting the sample data set
fig, ax = plt.subplots(dpi=200)
ax.plot(x_input, y_output, 'o')
plt.show()

# Define number of units for input layer  
m1 = 1
n1 = 6 

# Define number of units for hidden and output layers 
mh = n1
n4 = 1 

# Create a feed-forward neural network
fnn = nn.Sequential(nn.Linear(m1, n1),
                  nn.ReLU(),
                  nn.Linear(mh, mh),
                  nn.ReLU(),
                  nn.Linear(mh, mh),
                  nn.ReLU(),                 
                  nn.Linear(mh, n4))

# Initialize the optimizer  
optimizer = torch.optim.SGD(fnn.parameters(), lr=0.01) # lr is the learning rate

# Create a loss function
mse_loss = nn.MSELoss()

# Create a list to save the history of the loss for visualization later
training_loss = []

# Iterate the optimizer.  
for i in range(10000):
    # This is essential for the optimizer to keep
    # track of the gradients correctly
    # It is using some buffers internally that need to
    # be manually zeroed on each iteration.
    optimizer.zero_grad()
    
    # Make predictions
    y_pred = fnn(x_input)
    
    # Evaluate the loss - That's what you are minimizing
    loss = mse_loss(y_output, y_pred)
    
    # Evaluate the derivative of the loss with respect to all parameters 
    loss.backward()
    
    # And now we are ready to make a step
    optimizer.step()
    
    # Save the training loss for visualization later
    training_loss.append(loss.item())
    
    # Print the loss every 1000 iterations:
    if i % 1000 == 0:
        print('it = {0:d}: loss = {1:1.3f}'.format(i, loss.item()))

# Plotting the results 
fig, ax = plt.subplots(dpi=200)
ax.plot(x_input, y_output, 'o');
xx = torch.linspace(x_input.min(), x_input.max(), 100)[:, None]
yy = fnn(xx).detach().numpy()
ax.plot(xx, yy)
plt.show()

# Plotting the loss 
fig, ax = plt.subplots(dpi=200)
ax.plot(training_loss, label='Gradient descent loss')
ax.set_xlabel('Iteration')
ax.set_ylabel('Training loss')
plt.legend(loc='best', frameon=False)
plt.show()
