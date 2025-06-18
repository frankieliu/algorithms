
import numpy as np
from sklearn.datasets import make_moons 
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1/(1+np.exp(x))

class NN():
    def __init__(self, input_dim, hidden_dim, output_dim, random_seed=42):
        np.rand.manual_seed(random_seed)
        self.linear1 = np.rand.randn((input_dim, hidden_dim)) * 0.01
        self.bias1 = np.zeros((1,hidden_dim,))
        self.linear2 = np.rand.randn((hidden_dim, output_dim)) * 0.01
        self.bias2 = np.zeros((1,output_dim)) 
    def forward(self, x):
        self.z1 = np.einsum('bi, bij -> bj', x, self.linear1) + self.bias1
        self.a1 = sigmoid(self.zi)
        self.z2 = np.einsum('bi, bij -> bj', self.a1, self.linear2) + self.bias2
        self.a2 = sigmoid(self.z2)
        return self.a2
    def backward(self, X, y, learning_rate):
        m = X.size(0)
        self.Z2 = self.a2 - y
        self.W2 = (1/m)*np.einsum('bi, bj -> ijj', self.a1, self.Z2)/ self. 
        self.b2 = 
    def predict(self):
        pass

def loss_fn(ypred, y):
    return np.mean(-y*np.log(ypred) - (1-y)*np.log(1-ypred))

x,y = make_moons(n_samples = 1000, random_state = 42)
y = np.reshape(-1,1)
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.3)

model = NN()
epochs = 10000
for epoch 
ypred = model.forward(x)
loss = loss_fn(ypred, y)
model.backward(x)