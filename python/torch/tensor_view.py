import torch
t = torch.rand(4,4)

# There is no copy here
b = t.view(2,8)
print(b)
print(t)

base = torch.tensor([[0,1],[2,3]])
t = base.transpose(0, 1)
t.is_contiguous()

# continguous means if row order
# form, transposing will cause
# access to jump across rows

# causes a copy to be made, since
# the .transpose initally just created
# a view
t.contiguous()