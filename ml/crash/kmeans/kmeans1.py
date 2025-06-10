import numpy as np
class KMeans:

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=42):
        self.k = n_clusters
        np.random.seed(random_state)
        # self.rng = np.random.default_rng(seed=random_state)
        self.max_iter, self.tol = max_iter, tol
    def init(self, X):
        # idx = self.rng.permutation(X.shape[0])
        idx = np.random.permutation(X.shape[0])
        return X[idx[:self.k]]
    def update_labels(self, X, centroids):
        dist = np.sum((X[:,np.newaxis,:] - centroids)**2, axis=2)
        return np.argmin(dist,axis=1)
    def update_centroids(self, X, labels):
        centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            centroids[i] = np.mean(X[labels == i], axis = 0)
        return centroids
    def fit(self, X):
        X = np.array(X)
        centroids = self.init(X) 
        for i in range(self.max_iter):
            labels = self.update_labels(X, centroids)
            new_centroids = self.update_centroids(X, labels)
            print(new_centroids)
            shift = np.sqrt(np.sum((new_centroids - centroids)**2, axis = 1))
            if np.all(shift < self.tol):
                self.centroids = new_centroids
                self.labels = labels
                return self
            centroids = new_centroids