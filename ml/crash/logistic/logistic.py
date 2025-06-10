import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Linear model
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply sigmoid function
            y_predicted = self._sigmoid(linear_model)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict_prob(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_prob(X) >= threshold).astype(int)

# Example usage
if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(42)
    X = np.random.randn(100, 2)  # 100 samples, 2 features
    y = np.array([0 if x1 + x2 < 0 else 1 for x1, x2 in X])  # Simple decision boundary
    
    # Split into train/test
    X_train, y_train = X[:80], y[:80]
    X_test, y_test = X[80:], y[80:]
    
    # Create and train model
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Print learned parameters
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias}")