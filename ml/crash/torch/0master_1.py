# %%
import torch

# %%
scalar = torch.tensor(7)
print(scalar)
print(scalar.size)
# %%
print("hello")
# %%
print(scalar)
# %% [markdown]
Does this work
That is great just inserting markdown here
# %%
scalar.ndim

# %%
scalar
# %%
scalar.item()
# %%
vect = torch.tensor([7,7])
# %%
vect.ndim
# %%
# error: item only
# works for scalar
vect.item()

# %%
vect.size()
# %%
mat = torch.tensor([[1,2],[3,4]])
# %%
# mispelling
mat.ndims
# %%
mat.ndim
# %%
mat.size()
# %%
# shape and size()
# give the same answer
mat.shape
# %%

mat = torch.tensor([[[1,2],[3,4],[5,6]]])
# %%
mat.size
# %%
mat.shape
# %%
mat[0]
# %%
mat[0][1]
# %%
mat = torch.rand(size=[1,3,3,])
# %%
mat
# %%
mat
# %%
mat.dtype
# %%
torch.arange(0, 10, 1)
# %%
torch.ones_like(torch.arange(0,10,1))
# %%
mat.device
# %%
torch.manual_seed(42)
# %%
linear = torch.nn.Linear(3, 2)
# %%
linear
# %%
a = torch.tensor([[1,2],[2,3],[3,4]], dtype=torch.float32)
linear(a.T)# %%

# a = torch.tensor([[1,2,3],[4,5,6]])
# linear(a)
# %%
a = torch.tensor(
    [
    [[1,2,3],[3,4,5]],
    [[2,3,1],[3,4,5]]
    ])

# %%
a
# %%
linear(a.type(torch.float32))
# %%
a = torch.arange(0,10)
# %%
a
# %%
len(a)
# %%
b = a.view(1, len(a))
# %%
b

# %%
len(b)
# %%
torch.stack([a,a,a], dim=1)

# %%
a
# %%
a = torch.arange(0,27)
b = a.reshape(3,3,3)
c = b + 27
# %%
c

# %%
torch.stack([b,c],dim=1)

# %%
a = torch.arange(0,10)
# %%
b = a+10
# %%
torch.stack([a,b],dim=1)

# %%
