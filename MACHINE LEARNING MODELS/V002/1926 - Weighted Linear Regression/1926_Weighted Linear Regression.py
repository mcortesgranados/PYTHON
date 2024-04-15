# File name: 1926_Weighted Linear Regression.py
# @author Manuela Cortes Granados - 14 Abril 2024 1:42 PM

"""
Weighted Linear Regression is a variation of linear regression where each data point is given a weight indicating its relative 
importance in the regression analysis. The weighted least squares method is commonly used for this purpose.

In this example:

1. We generate synthetic data X and y with a known linear relationship (y = true_slope * X + true_intercept + noise).

2. We generate random weights for each data point, indicating their relative importance in the regression analysis.

3. We fit a weighted linear regression model using the LinearRegression class from scikit-learn. We pass the weights to the sample_weight parameter of the fit method.

4, Finally, we extract the estimated slope and intercept from the fitted model.

This example demonstrates how to perform Weighted Linear Regression using scikit-learn, a popular library for machine learning tasks in Python.

"""

import numpy as np
from sklearn.linear_model import LinearRegression

# Generate synthetic data
np.random.seed(0)
X = np.linspace(0, 10, 100).reshape(-1, 1)  # Reshape to create a column vector
true_slope = 2
true_intercept = 1
y = true_slope * X + true_intercept + np.random.normal(scale=2, size=X.shape)

# Generate weights (random in this example)
weights = np.random.rand(len(X))

# Fit weighted linear regression model
weighted_lr = LinearRegression()
weighted_lr.fit(X, y, sample_weight=weights)

# Extract estimated coefficients
intercept = weighted_lr.intercept_
slope = weighted_lr.coef_[0]

print(f"Estimated Slope: {slope}, Estimated Intercept: {intercept}")
