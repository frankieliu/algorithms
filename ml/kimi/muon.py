import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_vectors(ax, vectors, colors=None, labels=None):
    """Helper function to plot vectors from the origin."""
    for i, vec in enumerate(vectors):
        color = colors[i] if colors else 'b'
        label = labels[i] if labels else f'Vector {i+1}'
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=color, length=1,
                  normalize=False, arrow_length_ratio=0.1, label=label)

def plot_unit_sphere(ax):
    """Helper function to plot a unit sphere."""
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.1)

def plot_ellipsoid(ax, matrix, color='blue', alpha=0.3):
    """Helper function to plot the transformation of a unit sphere."""
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    sphere_points = np.vstack([x.ravel(), y.ravel(), z.ravel()])
    transformed_points = matrix @ sphere_points
    tx = transformed_points[0].reshape(x.shape)
    ty = transformed_points[1].reshape(y.shape)
    tz = transformed_points[2].reshape(z.shape)
    ax.plot_wireframe(tx, ty, tz, color=color, alpha=alpha)


# 1. Generate a random 3x3 symmetric matrix
# A = Q D Q^T where Q is orthogonal and D is diagonal
np.random.seed(42) # for reproducibility

# Generate random orthogonal matrix Q
Q, _ = np.linalg.qr(np.random.rand(3, 3))

# Generate random positive eigenvalues (singular values for symmetric matrix)
# Let's make them somewhat distinct to see the effect clearly
eigenvalues = np.diag(np.sort(np.random.rand(3) * 5 + 1)[::-1]) # Ensure positive and sorted descending
# eigenvalues = np.diag([3, 2, 0.5]) # Example fixed eigenvalues for clarity

A = Q @ eigenvalues @ Q.T

print("Original Symmetric Matrix A:")
print(A)
print("\nOriginal Eigenvalues (Singular Values):")
print(np.diag(eigenvalues))

# Calculate eigenvectors (which are also singular vectors for symmetric matrix)
eigenvalues_A, eigenvectors_A = np.linalg.eig(A)

print("\nEigenvalues of A (sorted):", np.sort(eigenvalues_A)[::-1])
print("Eigenvectors of A (columns):")
print(eigenvectors_A)

# Ensure eigenvectors are consistently oriented for plotting (optional but good practice)
for i in range(eigenvectors_A.shape[1]):
    if eigenvectors_A[0, i] < 0:
        eigenvectors_A[:, i] *= -1

# 2. Plot original eigenvectors and transformed unit sphere
fig = plt.figure(figsize=(18, 8))

ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title('Original Matrix A: Eigenvectors and Transformed Unit Sphere')
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
ax1.set_xlim([-4, 4]); ax1.set_ylim([-4, 4]); ax1.set_zlim([-4, 4])
ax1.view_init(elev=20, azim=45)

plot_unit_sphere(ax1)
plot_vectors(ax1, eigenvectors_A.T, colors=['r', 'g', 'b'], labels=['v1', 'v2', 'v3'])
plot_ellipsoid(ax1, A) # Plot the transformed unit sphere

# Add text labels for eigenvectors
for i, vec in enumerate(eigenvectors_A.T):
    ax1.text(vec[0]*1.2, vec[1]*1.2, vec[2]*1.2, f'v{i+1}', color=['r', 'g', 'b'][i], fontsize=10)

ax1.legend()


# 3. Simulate "Newton-Schulz like" normalization of eigenvalues/singular values
# In the context of Muon, this would be applied to the gradient's singular values,
# effectively making the update more "orthogonal" or "balanced."
# Here, we'll simulate applying a function that pushes singular values towards 1.
# Let's use a simplified function: f(s) = s / sqrt(s^2 + epsilon) or simply capping/scaling
# A common simplified idea is to push singular values toward 1.
# Here, let's normalize them by the largest singular value, or cap them.

# A simple "normalization" effect: Push all non-zero singular values towards 1
# This is NOT the exact Newton-Schulz iteration but illustrates the "leveling" idea.
# For simplicity, let's set them all to 1 for non-zero, or some fixed value.
normalized_eigenvalues_values = np.array([1.0 if val > 0 else 0 for val in np.diag(eigenvalues)])
normalized_eigenvalues_matrix = np.diag(normalized_eigenvalues_values)

# Reconstruct a "normalized" matrix based on the original eigenvectors
# and these normalized singular values (eigenvalues in this symmetric case)
A_normalized = Q @ normalized_eigenvalues_matrix @ Q.T

print("\n'Normalized' Eigenvalues (pushed towards 1):")
print(np.diag(normalized_eigenvalues_matrix))
print("\n'Normalized' Matrix A_normalized (reconstructed with normalized singular values):")
print(A_normalized)


# Calculate eigenvectors of the "normalized" matrix (they should be the same directions)
eigenvalues_norm, eigenvectors_norm = np.linalg.eig(A_normalized)
# Ensure eigenvectors are consistently oriented for plotting
for i in range(eigenvectors_norm.shape[1]):
    if eigenvectors_norm[0, i] < 0:
        eigenvectors_norm[:, i] *= -1

ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title("Matrix with 'Normalized' Singular Values: Eigenvectors and Transformed Unit Sphere")
ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
ax2.set_xlim([-4, 4]); ax2.set_ylim([-4, 4]); ax2.set_zlim([-4, 4])
ax2.view_init(elev=20, azim=45)

plot_unit_sphere(ax2)
# Eigenvectors remain in the same directions, only their scaling for transformation changes
plot_vectors(ax2, eigenvectors_norm.T, colors=['r', 'g', 'b'], labels=['v1', 'v2', 'v3'])
plot_ellipsoid(ax2, A_normalized, color='orange') # Plot the transformed unit sphere with normalized values

# Add text labels for eigenvectors
for i, vec in enumerate(eigenvectors_norm.T):
    ax2.text(vec[0]*1.2, vec[1]*1.2, vec[2]*1.2, f'v{i+1}', color=['r', 'g', 'b'][i], fontsize=10)

ax2.legend()
plt.tight_layout()
plt.show()

print("\n--- Observation ---")
print("1. Eigenvectors (singular vectors) represent the principal directions of stretching/compression.")
print("2. For a symmetric matrix, eigenvalues are its singular values.")
print("3. In the 'Original Matrix A', the ellipsoid is stretched more along the eigenvector corresponding to the largest eigenvalue, and less along the one with the smallest.")
print("4. When we 'normalize' the eigenvalues (by pushing them towards 1, similar to what Newton-Schulz aims for singular values), the *directions* of the eigenvectors remain the same.")
print("5. However, the *shape* of the transformed ellipsoid changes. It becomes more spherical (less stretched/compressed along specific axes), reflecting a more 'balanced' transformation. This is analogous to how Muon's updates become more balanced across different singular value directions.")
print("6. This visual demonstrates how the Newton-Schulz iteration, by acting on singular values, aims to make the transformation (or in Muon's case, the gradient update) more 'uniform' or 'orthogonal' in its scaling effect.")