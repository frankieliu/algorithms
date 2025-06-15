#%%
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
#%%
traind,testd = [
datasets.FashionMNIST(
    root="data",
    train=x,
    download=True,
    transform=ToTensor()
) for x in [True, False]]
#%%
trainl,testl = [
    DataLoader(x, batch_size=64)
    for x in [traind, testd] 
]
#%%
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.sequential = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10))
    def forward(self, x):
        x = self.flatten(x)
        return self.sequential(x)
#%%
device = "cpu"
model = NN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
#%%
def train(dl,loss_fn,optimizer):
    model.train() 
    for di, (x,y) in enumerate(dl):
        x,y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if di % 100 == 1:
            print(f"loss: {loss} [{di}/{len(dl)}]")

def test(dl,loss_fn):
    model.eval()
    loss,count = 0,0
    for di, (x,y) in enumerate(dl):
        x,y = x.to(device), y.to(device)
        y_pred = model(x)
        loss += loss_fn(y_pred, y).item()
        count += (y_pred.argmax(1) == y).type(torch.float).sum().item()
    print(f"loss: {loss} acc: {count/len(dl.dataset)}")
#%%
epochs = 5
for i in range(epochs):
    print(f"Epoch {i}")
    train(trainl, loss_fn, optimizer)
    test(testl, loss_fn)
#%%
torch.save(model.state_dict(), "model.pth")
model1 = NN().to(device)
model1.load_state_dict(torch.load("model.pth"))
#%%
X,y = testd[0]
pred = model1(X)
y_pred = pred.argmax(1).item()
print(f"pred: {y_pred} target: {y}")
plt.imshow(X[0],cmap="grey")