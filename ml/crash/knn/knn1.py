import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k):
        self.k = k
    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
    def predict_one(self, x):
        distances = np.sqrt(np.sum((self.X - x)**2, axis=1))
        nei = np.argpartition(distances, self.k)[:self.k]
        return Counter(self.y[nei]).most_common(1)[0][0]
    def predict(self, X):
        return [self.predict_one(np.array(x)) for x in X]

