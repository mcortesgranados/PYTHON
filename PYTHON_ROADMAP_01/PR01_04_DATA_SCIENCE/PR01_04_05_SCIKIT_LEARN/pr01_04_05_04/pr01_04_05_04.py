"""
04. Dimensionality Reduction: Reducing the number of features while preserving important information.

Explanation:

Importing Libraries: We import the necessary libraries including numpy for numerical computations, matplotlib for visualization, and scikit-learn for dimensionality reduction.

Loading Data: We load the digits dataset from scikit-learn's datasets module. This dataset contains images of handwritten digits.

Initializing PCA: We initialize a PCA (Principal Component Analysis) object with 2 principal components. PCA is a popular technique for dimensionality reduction.

Fitting and Transforming Data: We fit the PCA model to the data and then transform the data to reduce its dimensions. This step projects the original high-dimensional data onto a lower-dimensional space while preserving important information.

Visualizing Reduced Data: We visualize the reduced data by plotting the first two principal components against each other. Each point in the plot represents a data point in the reduced space, with colors representing different digit labels.

Plotting: We plot the reduced data using matplotlib, labeling the axes and adding a colorbar to show the digit labels.

"""

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Initialize PCA with 2 principal components
pca = PCA(n_components=2)

# Fit and transform the data to reduce dimensions
X_pca = pca.fit_transform(X)

# Visualize the reduced data
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Dimensionality Reduction')
plt.colorbar(label='Digit Label')
plt.show()
