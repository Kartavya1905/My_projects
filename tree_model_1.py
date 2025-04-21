# For AI 681 students only
# Tree-based method

import numpy as np

# Sample data
X = np.array([[2.5], [5.2], [7.3], [8.4], [1.3], [4.1], [6.2], [3.5]])
y = np.array([3.2, 4.5, 7.4, 8.8, 1.5, 4.0, 6.5, 3.3])

print(X)
print(f'y: {y}')

# Calculate RSS  = N * variance with a check for empty arrays
def rss(y):
    if len(y) == 0:
        return 0
    return np.var(y) * len(y)

i = 1 

# Find the best split
def find_best_split(X, y):
    best_var = float('inf')    # positive infinity
    best_split = None

    global i
    
    num_features = range(X.shape[1])
 
    # loop through each feature 
    for feature_index in num_features:

        unique_features = np.unique(X[:, feature_index])
        if i == 1:
            print(f'Unique features: {unique_features}') 
        # loop through each data point of the current feature 
        for split_value in unique_features:
            if i == 1:
                print(split_value)
            # split the dataset     
            left_mask = X[:, feature_index] <= split_value
            if i == 1:
                print(f'left_mask: {left_mask}') 
                
            right_mask = ~left_mask
            if i == 1:
                print(f'right_mask: {right_mask}')
                
            # Check if either split is empty
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            # Compute RSS for each branch after the split 
            left_var = rss(y[left_mask])
            right_var = rss(y[right_mask])
            
            if i == 1:
                print(y[left_mask])
                print(f'left_var: {left_var}')
                print(y[right_mask])
                print(f'right_var: {right_var}')
                i = 2
                
            # Compute total RSS after the split             
            total_var = left_var + right_var

            # Save the split if it is the best so far 
            if total_var < best_var:
                best_var = total_var
                best_split = (feature_index, split_value)
    return best_split

# Decision Tree Node
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
 
                
# Build Tree
def build_tree(X, y, depth=0, max_depth=3):
 
    if len(set(y)) == 1 or depth == max_depth:
        return Node(value=np.mean(y))
    
    # find the best split and its feature 
    feature_index, split = find_best_split(X, y)
    
    print(feature_index, split )
    
    if feature_index is None:
        return Node(value=np.mean(y))
    
    left_mask = X[:, feature_index] <= split
    right_mask = ~left_mask

    print(f'left_mask on tree: {left_mask}' )
    
    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:  # Check again for empty splits
        return Node(value=np.mean(y))
    
    left = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth)
    right = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth)
     

    return Node(feature_index, split, left, right)

# Prediction
def predict_tree(node, x):
    if node.value is not None:
        return node.value
    
    if x[node.feature_index] <= node.threshold:
        return predict_tree(node.left, x)
    else:
        return predict_tree(node.right, x)
		
# Build the tree and predict
print('===================================')
print('Building a Tree')
print('===================================')
tree = build_tree(X, y, max_depth=3)

predictions = [predict_tree(tree, x) for x in X]

print("Predictions:", predictions)
