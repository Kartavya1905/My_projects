# For AI 681 students only
# Tree-based method 

from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
class_names = data.target_names
 
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Train a single decision tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42)   
tree.fit(X_train, y_train)
acc_tree = accuracy_score(y_test, tree.predict(X_test))
  
print(f"Test Accuracy of the Decision Tree: {acc_tree:.3f}")
 
# Plot the tree
plt.figure(figsize=(20, 10))
plot_tree(tree, 
          feature_names=feature_names, 
          class_names=class_names, 
          filled=True, 
          rounded=True,
          fontsize=12)
plt.title("Decision Tree (max_depth=3)")
plt.savefig("treeplot.pdf", dpi=300)
plt.show()
