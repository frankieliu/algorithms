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
