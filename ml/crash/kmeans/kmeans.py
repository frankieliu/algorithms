import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        """
        Initialize KMeans clustering algorithm.
        
        Parameters:
        - n_clusters: Number of clusters (default: 3)
        - max_iter: Maximum number of iterations (default: 300)
        - tol: Tolerance for convergence (default: 1e-4)
        - random_state: Seed for random initialization (default: None)
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        
    def _initialize_centroids(self, X):
        """Randomly initialize centroids from the data points."""
        np.random.seed(self.random_state)
        indices = np.random.permutation(X.shape[0])
        centroids = X[indices[:self.n_clusters]]
        return centroids
    
    def _assign_clusters(self, X):
        """Assign each data point to the nearest centroid."""
        distances = np.sqrt(((X[:, np.newaxis, :] - self.centroids) ** 2).sum(axis=2))
        self.labels = np.argmin(distances, axis=1)
        
    def _update_centroids(self, X):
        """Update centroids as the mean of assigned points."""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            new_centroids[i] = np.mean(X[self.labels == i], axis=0)
        return new_centroids
    
    def fit(self, X):
        """
        Fit the k-means model to the data.
        
        Parameters:
        - X: Input data, numpy array of shape (n_samples, n_features)
        """
        # Initialize centroids
        self.centroids = self._initialize_centroids(X)
        
        for _ in range(self.max_iter):
            # Assign points to nearest centroid
            self._assign_clusters(X)
            
            # Update centroids
            new_centroids = self._update_centroids(X)
            
            print(new_centroids)
            # Check for convergence
            centroid_shift = np.sqrt(((new_centroids - self.centroids) ** 2).sum(axis=1))
            if np.all(centroid_shift < self.tol):
                break
                
            self.centroids = new_centroids
            
        return self
    
    def predict(self, X):
        """
        Predict the closest cluster for each sample in X.
        
        Parameters:
        - X: Input data, numpy array of shape (n_samples, n_features)
        
        Returns:
        - labels: Index of the cluster each sample belongs to
        """
        if self.centroids is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        distances = np.sqrt(((X[:, np.newaxis, :] - self.centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)