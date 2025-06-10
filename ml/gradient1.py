import numpy as np
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s*(1-s)
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    def loss_fn(self, y, y_hat):
        return = -np.mean(
            y * np.log(y_hat) +
            (1-y) * np.log(1-y_hat), axis=0)
    def backward(self, X, y, learning_rate):
        # a2->z2->a1->z1
        
        dZ2 = self.a2 - y