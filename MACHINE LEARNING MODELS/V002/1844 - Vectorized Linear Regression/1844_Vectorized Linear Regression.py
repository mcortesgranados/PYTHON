# File name: 1844_Vectorized Linear Regression.py
# @author Manuela Cortes Granados - 15 Abril 2024 2:42 AM

import numpy as np

# Generate synthetic data
np.random.seed(0)
X = np.linspace(0, 10, 100).reshape(-1, 1)  # Reshape to create a column vector
true_slope = 2
true_intercept = 1
y = true_slope * X + true_intercept + np.random.normal(scale=2, size=X.shape)

# Add a column of ones to X for the intercept term
X_with_intercept = np.hstack((np.ones_like(X), X))

# Perform vectorized linear regression
coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y

# Extract slope and intercept from coefficients
intercept = coefficients[0]
slope = coefficients[1]

print(f"Estimated Slope: {slope}, Estimated Intercept: {intercept}")
