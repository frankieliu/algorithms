import numpy as np

I, J = 3, 2
A = np.random.rand(I, J, I, I, J)

# Einsum version
out = np.einsum('ijiij->j', A)

# Manual version
out_manual = np.zeros(J)
for j in range(J):
    for i in range(I):
        out_manual[j] += A[i, j, i, i, j]

print(out)
print(out_manual)