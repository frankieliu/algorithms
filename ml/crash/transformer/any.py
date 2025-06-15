#%%
import torch

# %%
# 4 x 2 x 3 = 24
a = torch.arange(24)
a = a.view(4, 2, 3)
print(a)
# %%
for i in range(23):
    mask = (a == i).any(dim=-1)
    print(i, mask, a[mask])

# %%

# Sample tensors
tensor1 = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
tensor2 = torch.tensor([[10, 20, 30, 40, 50], [60, 70, 80, 90, 100]])

# Value to search for
value_to_find = 8

# Find the index where tensor1 matches the value
where = torch.where(tensor1 == value_to_find)
index = where

# Apply the index to tensor2
result_tensor = tensor2[..., index]

print("Original tensors:")
print("Where: ", where)
print("Index: ", index)
print("Tensor 1:", tensor1)
print("Tensor 2:", tensor2)
print("\nResult tensor:", result_tensor)  # Output: tensor([[80]])
# %%
idx, values = torch.where(
    torch.tensor(
        [
            [0.6, 0.0, 0.0, 0.0],
            [0.0, 0.4, 0.0, 0.0],
            [0.0, 0.0, 1.2, 0.0],
            [0.0, 0.0, 0.0, -0.4],
        ]
    )
)

print(idx)
print(values)
#%%

hidden_dim = 128
hidden_states = torch.randn(6, hidden_dim)  # 6 tokens

# Case 1: Empty selection (no tokens use this expert)
top_x = torch.tensor([], dtype=torch.long)
print(hidden_states[top_x].shape)                    # torch.Size([0, 128])
print(hidden_states[None, top_x].shape)              # torch.Size([1, 0, 128])
print(hidden_states[None, top_x].reshape(-1, hidden_dim).shape)  # torch.Size([0, 128])

# Case 2: Non-empty selection
top_x = torch.tensor([0, 2], dtype=torch.long)
print(hidden_states[top_x].shape)                    # torch.Size([2, 128])
print(hidden_states[None, top_x].reshape(-1, hidden_dim).shape)  # torch.Size([2, 128])


# %%
a = torch.rand((100,))>0.5
a = a.reshape(2,5,10)
print(a)
first, second, third = torch.where(a)
print(first, second, third)
print(a.type(torch.float).sum())
print(len(first))
b = torch.zeros_like(a)
# a 2,5,10
# b.scatter_() --> will only scatter in one dimension
# note scatter_ is in place, which scatter is not
#%%
import torch

# Create an initial 3D tensor
input_tensor = torch.zeros(3, 4, 5)
print("Initial input_tensor:")
print(input_tensor)
print(input_tensor.type())

# Create a source tensor with values to scatter
src_tensor = torch.arange(1, 13).reshape(3, 4).type(torch.float)
print("\nSource tensor:")
print(src_tensor)

# Create an index tensor to specify where to scatter the values along a dimension
# The index tensor determines the target index in the specified dimension (`dim`) for each element in the source tensor.
# For `dim=2`, the index tensor determines the target column index in the 3rd dimension.
index_tensor = torch.tensor([[0, 2, 1, 3],
                            [1, 0, 3, 2],
                            [2, 3, 0, 1]])
print("\nIndex tensor:")
print(index_tensor)

# Scatter the values from src_tensor into input_tensor along the 3rd dimension (dim=2)
# The scatter_ operation is explained as follows for a 3D tensor:
# For `dim=2`, input_tensor[i][j][index_tensor[i][j]] = src_tensor[i][j]
input_tensor.scatter_(2, index_tensor.unsqueeze(-1), src_tensor.unsqueeze(-1))  # We unsqueeze to make the index and source tensors have the same number of dimensions as the input tensor, which is a requirement for scatter_.

print("\nTensor after scatter_:")
print(input_tensor)
#%%
import torch

# Create a 3D tensor
src = torch.arange(1, 28).reshape(3, 3, 3) 
# tensor([[[ 1,  2,  3],
#          [ 4,  5,  6],
#          [ 7,  8,  9]],
#
#         [[10, 11, 12],
#          [13, 14, 15],
#          [16, 17, 18]],
#
#         [[19, 20, 21],
#          [22, 23, 24],
#          [25, 26, 27]]])

# Create an index tensor to scatter along the second dimension (dim=1)
# The index tensor should have the same dimensions as src
# The values in index specify the target indices in the specified dimension
index = torch.tensor([[[0, 1, 0],
                       [2, 0, 1],
                       [1, 2, 2]],

                      [[1, 0, 2],
                       [0, 2, 1],
                       [2, 1, 0]],

                      [[2, 2, 1],
                       [1, 0, 0],
                       [0, 1, 2]]])
# The index values must be within the bounds of the target dimension size
# In this case, the second dimension has a size of 3, so indices can be 0, 1, or 2

# Create a destination tensor to scatter into
# The destination tensor should have the same shape as src, but the size of the scattered dimension can differ
# Here, we keep the same shape for simplicity
out = torch.zeros_like(src)

# Scatter values from src into out based on index along dim=1
# The scatter_ operation performs an in-place scatter
out.scatter_(dim=1, index=index, src=src)

print("Source tensor (src):")
print(src)
print("\nIndex tensor:")
print(index)
print("\nOutput tensor (out) after scatter along dim=1:")
print(out)


#%%
# Example tensor
x = torch.tensor([[4, 9, 7],
                  [2, 6, 1],
                  [8, 3, 5]])

# Find the indices and values of the top 2 elements in each row
top_values, top_indices = torch.topk(x, 2, dim=1)

print("Top values:\n", top_values)
print("Top indices:\n", top_indices)
#%%

# Using index_add_
tensor = torch.zeros(5)
indices = torch.tensor([1, 3, 1, 1])  # Note duplicate indices
values = torch.tensor([1., 2., 3., 4.])
tensor.index_add_(0, indices, values)
print(tensor)  # tensor([0., 8., 0., 2., 0.]) - values at index 1 are accumulated

# Using indexing (incorrect for duplicates)
tensor = torch.zeros(5)
tensor[indices] += values
print(tensor)  # tensor([0., 4., 0., 2., 0.]) - only last value for index 1 is kept
#%%
# Create a tensor with dimensions (2, 1, 3)
x = torch.tensor([[[1, 2, 3]], [[4, 5, 6]]])
print(f"Original tensor:\n {x}")
print(f"Original size: {x.size()}")

# Expand the second dimension to size 4
y = x.expand(2, 4, 3)
print(f"Expanded tensor:\n {y}")
print(f"Expanded size: {y.size()}")