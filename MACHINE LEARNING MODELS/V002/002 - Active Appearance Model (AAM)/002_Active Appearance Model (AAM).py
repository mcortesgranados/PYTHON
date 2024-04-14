# This is a Python file created by the program.
# File name: 002_Active Appearance Model (AAM).py
# @author Manuela Cortes Granados - 14 Abril 2024 1:42 PM


"""

Implementing a full Active Appearance Model (AAM) in Python would require several steps,
 including data preprocessing, feature extraction, model training, and deployment. 
 Given the complexity of the AAM model, providing a complete implementation in this format is not feasible. However
 , I can give you a simplified example that demonstrates the basic concept of building an AAM using a library like
 scikit-image.

"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.feature import local_binary_pattern
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Load example image
image = data.astronaut()

# Resize image for simplicity
image = resize(image, (200, 200))

# Convert image to grayscale
gray_image = np.mean(image, axis=2)

# Extract Local Binary Pattern features
radius = 3
n_points = 8 * radius
lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')

# Apply PCA for dimensionality reduction
n_components = min(gray_image.shape) // 2  # Adjust based on the image size
pca = PCA(n_components=n_components)
lbp_1d = np.ravel(lbp)
lbp_pca = pca.fit_transform(lbp_1d.reshape(-1, 1))

# Fit linear regression model
X = np.arange(n_components).reshape(-1, 1)
y = lbp_pca[:, 0]  # Using first PCA component as target
model = LinearRegression()
model.fit(X, y)

# Generate new appearance model based on linear regression prediction
predicted_pca = model.predict(X)
predicted_lbp = pca.inverse_transform(predicted_pca)
predicted_lbp = np.clip(predicted_lbp, 0, 1)
predicted_lbp = np.reshape(predicted_lbp, gray_image.shape)

# Plot original and predicted images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(predicted_lbp, cmap='gray')
plt.title('Predicted Image')
plt.axis('off')

plt.show()
