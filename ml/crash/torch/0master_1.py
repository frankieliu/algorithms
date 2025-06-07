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
vect.item()
# %%
vect.size()
# %%
mat = torch.tensor([[1,2],[3,4]])
# %%
mat.ndims
# %%
mat.ndim
# %%
mat.size()
# %%
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
ma
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
