"""
03. Clustering: Grouping similar data points together based on features.

Explanation:

Importing Libraries: We import the necessary libraries including numpy for numerical computations, matplotlib for visualization, and scikit-learn for clustering.

Generating Data: We generate synthetic data for clustering using the make_blobs function from scikit-learn. This function creates a dataset with specified number of samples, centers, cluster standard deviation, and random state.

Initializing Model: We initialize a KMeans clustering model using the KMeans class from scikit-learn. We specify the number of clusters to be formed as a parameter.

Fitting Model: We fit the KMeans model to the data using the fit method. This step identifies the cluster centers and assigns each data point to the nearest cluster.

Predicting Labels: We predict the cluster labels for each data point using the predict method.

Visualizing Clusters: We visualize the clusters by plotting the data points with their assigned cluster labels. Additionally, we plot the centroids of the clusters as black points.

Plotting: We plot the clusters along with their centroids using matplotlib.
"""

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate synthetic data for clustering
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Initialize the KMeans clustering model
kmeans = KMeans(n_clusters=4)

# Fit the model to the data
kmeans.fit(X)

# Predict the cluster labels for each data point
y_kmeans = kmeans.predict(X)

# Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Plotting the centroids of the clusters
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KMeans Clustering')
plt.show()
