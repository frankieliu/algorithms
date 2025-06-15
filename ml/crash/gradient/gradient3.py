import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
class NN:
    def __init__(
            self, 
            input_size, hidden_size, output_size,
            random_seed = 42):
        np.random.seed(random_seed)
        self.W1 = np.random.randn(
            input_size, hidden_size) * 0.01
        self.b1 = np.zeros(
            (1, hidden_size))
        self.W2 = np.random.randn(
            hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, X):
        return 1/(1+np.exp(-X))
    def sigmoid_derivative(self, X):
        s = 1/(1+np.exp(-X))
        return s*(1-s)
    def forward(self,X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def loss_fn(self, y, y_hat):
        return -np.mean(
            y*np.log(y_hat) +
            (1-y)*np.log(1-y_hat))
    def backward(self, X, y, learning_rate):
        m = X.shape[0]  
        dZ2 = self.a2 - y
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        dZ1 = np.dot(dZ2, self.W2.T) * self.sigmoid_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    def train(self, X, y, epochs=10000, learning_rate=0.1, verbose=True):
        losses = []
        for i in range(epochs):
            y_hat = self.forward(X) 
            loss = self.loss_fn(y, y_hat) 
            self.backward(X, y, learning_rate)
            losses.append(loss)
            if verbose and i % 1000 == 0:
                print(f"Epoch {i}  Loss {loss}")
        return losses
    def predict(self, X, threshold=0.5):
        y_hat = self.forward(X)
        return (y_hat > threshold).astype(int)

X, y = make_moons(n_samples = 1000, noise=0.2, random_state=42)
y = y.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean)/std
X_test = (X_test - mean)/std
nn = NN(2, 4, 1)
losses = nn.train(X_train, y_train)

#plt.plot(losses)
#plt.show()

y_pred = nn.predict(X_test)
acc = np.mean(y_test == y_pred)
print(f"Acc: {acc}")

def plot_boundary(model, X, y):
    xmin, xmax, ymin, ymax = (
        X[:,0].min()-1, X[:,0].max()+1,
        X[:,1].min()-1, X[:,1].max()+1
    )
    h = 0.01
    xx, yy = np.meshgrid(
        np.arange(xmin, xmax, h),
        np.arange(ymin, ymax, h))
    Z = model.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z,cmap=plt.cm.Spectral)
    plt.scatter(X[:,0],X[:,1],c=y.ravel(), cmap=plt.cm.Spectral)
    plt.show()

plot_boundary(nn, X_test, y_test)