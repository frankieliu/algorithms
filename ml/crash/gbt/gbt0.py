from sklearn.tree import DecisionTreeRegressor
import numpy as np

class GradientBoostedTree():
    def __init__(self, max_trees=20,learning_rate=0.5, max_depth=2):
        self.max_trees = max_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth 
        self.initial_pred = None
        self.trees = None

    def fit(self, X, y):
        initial_pred = [np.mean(y)] * len(y)
        pred = initial_pred 
        trees = []
        for i in range(self.max_trees):
            residuals = y - pred
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            pred += tree.predict(X) * self.learning_rate
            trees.append(tree)
        self.initial_pred = initial_pred
        self.trees = trees

    def predict(self, X):
        prediction = self.initial_pred
        for i in range(len(self.trees)):
            prediction += self.trees[i].predict(X) * self.learning_rate
        return prediction
