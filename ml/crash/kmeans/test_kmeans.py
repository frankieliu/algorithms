import numpy as np
import matplotlib.pyplot as plt
from kmeans1 import KMeans

# Generate sample data
np.random.seed(42)
X = np.vstack([
    np.random.normal(loc=[0, 0], scale=1, size=(100, 2)),
    np.random.normal(loc=[5, 5], scale=1, size=(100, 2)),
    np.random.normal(loc=[-5, 5], scale=1, size=(100, 2))
])

# Create and fit KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get cluster labels and centroids
labels = kmeans.labels
centroids = kmeans.centroids

print("Cluster centroids:")
print(centroids)

# Visualization (optional, requires matplotlib)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3)
plt.title("K-Means Clustering")
plt.show()