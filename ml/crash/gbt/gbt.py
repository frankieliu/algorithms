from sklearn.tree import DecisionTreeRegressor
import numpy as np

def gradient_boosted_trees(X, y, num_trees, learning_rate, max_depth):
    """
    X: Input features (n_samples, n_features)
    y: Target values (n_samples,)
    num_trees: Number of boosting iterations (trees)
    learning_rate: Shrinkage factor for tree contributions
    max_depth: Maximum depth of each decision tree
    """
    
    # Initialize model with a constant value (mean of target)
    initial_prediction = np.mean(y)
    predictions = [initial_prediction] * len(y)  # Initial predictions
    
    trees = []  # To store the weak learners (trees)
    
    for t in range(num_trees):
        # Compute pseudo-residuals (negative gradient of loss function)
        residuals = y - predictions  # For MSE loss, residual = (y - prediction)
        
        # Fit a decision tree to the residuals (approximate the negative gradient)
        tree = DecisionTreeRegressor(max_depth=max_depth)
        tree.fit(X, residuals)
        
        # Update predictions by adding the tree's prediction (scaled by learning rate)
        tree_prediction = tree.predict(X)
        predictions += learning_rate * tree_prediction
        
        # Save the tree
        trees.append(tree)
    
    return trees, initial_prediction

def predict(X, trees, initial_prediction, learning_rate):
    """
    X: Input features to predict
    trees: List of trained decision trees
    initial_prediction: Starting prediction (mean(y))
    learning_rate: Same as in training
    """
    prediction = initial_prediction
    for tree in trees:
        prediction += learning_rate * tree.predict(X)
    return prediction
