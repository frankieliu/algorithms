# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# %%
train_data = datasets.FashionMNIST(
 root = "data",
 train = True,
 download = True,
 transform = ToTensor()
)
test_data = datasets.FashionMNIST(
 root = "data",
 train = False,
 download = True,
 transform = ToTensor()
)
# %%
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)
# %%
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.sequential = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.sequential(x)
# %%
device = "cpu"
loss_fn = nn.CrossEntropyLoss()
model = NN().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
# %%
print(model)
# %%
def train(data_loader, loss_fn, optimizer):
    size = len(data_loader.dataset)
    model.train()
    for i, (X,y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            loss = loss.item()
            print(f"Loss: {loss} [{(i+1)*len(X)}/{size}]")

def test(data_loader, loss_fn):
    sz = len(data_loader.dataset)
    model.eval()
    correct, loss = 0,0
    for i, (X,y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    print(f"Loss: {loss/len(data_loader)} Acc: {correct/sz}")

# %%
epoch = 5
for i in range(epoch):
    print(f"Epoch: {i+1}")
    train(train_loader, loss_fn, optimizer)
    test(test_loader, loss_fn)
