import math
import numpy as np
from collections import Counter


class TreeNode:
    def __init__(self, mclass=None):
        self.mclass = mclass
        self.left, self.right, self.feature, self.thresh = [None] * 4


class DecisionTreeClassifier:
    def __init__(self, max_depth=2, min_samples_split=2):
        self.tree = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.tree = self.build_tree(X, y, depth=0)

    def predict(self, X):
        return [self.predict_one(self.tree, x) for x in X]

    def predict_one(self, node, x):
        if node.mclass is not None:
            return node.mclass
        if x[node.feature] <= node.thresh:
            return self.predict_one(node.left, x)
        return self.predict_one(node.right, x)

    def build_tree(self, X, y, depth):
        nunique = len(np.unique(y))
        y_common = Counter(y).most_common(1)[0][0]
        nsamples = len(y)
        if nunique == 1 or depth == self.max_depth or nsamples < self.min_samples_split:
            return TreeNode(mclass=y_common)
        feature, thresh = self.best_split(X, y)
        if feature is None:
            return TreeNode(mclass=y_common)
        lmask = X[:, feature] <= thresh
        rmask = ~lmask
        node = TreeNode()
        node.feature, node.thresh = feature, thresh
        node.left = self.build_tree(X[lmask], y[lmask], depth + 1)
        node.right = self.build_tree(X[rmask], y[rmask], depth + 1)
        return node

    def best_split(self, X, y):
        _, nfeatures = X.shape
        best_score = math.inf
        feature, thresh = None, None
        for i in range(nfeatures):
            thresholds = np.unique(X[:, i])
            for t in thresholds:
                lmask = X[:, i] <= t
                rmask = ~lmask
                nl, nr = lmask.sum(), rmask.sum()
                sl, sr = self.gini(y[lmask]), self.gini(y[rmask])
                weighted = (nl * sl + nr * sr) / (nl + nr)
                if weighted < best_score:
                    feature, thresh = i, t
                    best_score = weighted
        return feature, thresh

    def gini(self, y):
        counts = np.bincount(y)
        counts = (counts / len(y)) ** 2
        return 1 - counts.sum()
