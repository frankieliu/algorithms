import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train, y_train):
        """Store the training data."""
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
    
    def predict(self, X_test):
        """Predict the class for test samples."""
        X_test = np.array(X_test)
        predictions = []
        
        for sample in X_test:
            # Calculate Euclidean distances using vectorized operations
            distances = np.sqrt(np.sum((self.X_train - sample)**2, axis=1))
            
            # Get indices of k nearest neighbors
            k_indices = np.argpartition(distances, self.k)[:self.k]
            
            # Get the labels of these neighbors
            k_nearest_labels = self.y_train[k_indices]
            
            # Find the most common label
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        
        return predictions
