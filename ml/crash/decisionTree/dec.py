import numpy as np
from collections import Counter

class TreeNode:
    def __init__(self, mclass=None, threshold=None, feature=None, left=None, right=None):
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.mclass = mclass

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria
        if (depth == self.max_depth) or (n_classes == 1) or (n_samples < self.min_samples_split):
            leaf_value = self._most_common_class(y)
            return TreeNode(mclass=leaf_value)

        # Find best split
        best_feature, best_threshold = self._best_split(X, y)

        if best_feature is None:
            leaf_value = self._most_common_class(y)
            return TreeNode(mclass=leaf_value)

        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Recursively build subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(
            threshold=best_threshold,
            feature=best_feature,
            left=left_subtree,
            right=right_subtree)
        
    def _best_split(self, X, y):
        best_gini = float('inf')
        best_feature, best_threshold = None, None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                left_gini = self._gini_impurity(y[left_mask])
                right_gini = self._gini_impurity(y[~left_mask])

                # Weighted Gini impurity
                n_left, n_right = len(y[left_mask]), len(y[~left_mask])
                weighted_gini = (n_left * left_gini + n_right * right_gini) / (n_left + n_right)

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _gini_impurity(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _most_common_class(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, node):
        if node.mclass is not None:
            return node.mclass
        else:
            if x[node.feature] <= node.threshold:
                return self._predict_tree(x, node.left)
            else:
                return self._predict_tree(x, node.right)