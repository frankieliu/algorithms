import numpy as np
from collections import Counter
import math

class TreeNode:
    def __init__(self, mclass=None):
        self.mclass = mclass
        self.feature, self.thresh, self.left, self.right = [None]*4
        
class DecisionTreeClassifier:
    def __init__(self, max_depth=2, min_samples_split=2): 
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    def fit(self, X, y):
        self.tree = self.build_tree(X,y,depth=0)
    def build_tree(self, X, y, depth):
        nclass = len(np.unique(y))
        # 1 returns the most common element which is a tuple (el, count)
        ycommon = Counter(y).most_common(1)[0][0]
        if nclass == 1 or depth == self.max_depth or len(y) < self.min_samples_split:
            return TreeNode(mclass = ycommon)
        feature, thresh = self.best_fit(X,y)
        if feature is None:
            return TreeNode(mclass = ycommon) 
        lmask = X[:,feature] <= thresh
        rmask = ~lmask
        left = self.build_tree(X[lmask], y[lmask], depth+1)
        right = self.build_tree(X[rmask], y[rmask], depth+1)
        node = TreeNode()
        node.feature, node.thresh = feature, thresh
        node.left, node.right = left, right
        return node

    def best_fit(self, X, y):
        _, nfeatures = X.shape
        feature, bthresh = None, None
        gbest = math.inf
        for i in range(nfeatures):
            thresholds = np.unique(X[:,i])
            for threshold in thresholds:
                lmask = X[:,i] <= threshold
                rmask = ~lmask
                nl, nr = lmask.sum(), rmask.sum()
                lscore = self.ginny(y[lmask])
                rscore = self.ginny(y[rmask])
                weighted = (nl*lscore + nr*rscore) / (nl+nr)
                if weighted < gbest:
                    gbest = weighted
                    feature = i
                    bthresh = threshold
        return feature, bthresh
    
    def ginny(self, y):
        count = np.bincount(y)
        count = (count / len(y)) ** 2
        return 1 - count.sum()

    def predict(self, X):
        return [self.predict_one(self.tree, x) for x in X]

    def predict_one(self, node: TreeNode, x):
        if node.mclass is not None:
            return node.mclass
        else:
            if x[node.feature] <= node.thresh:
                return self.predict_one(node.left, x)
            else:
                return self.predict_one(node.right, x) 