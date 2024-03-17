import numpy as np
import pandas as pd
import statsmodels.api as sm

# Generate sample data
np.random.seed(0)
X = np.random.rand(100, 2)
X = sm.add_constant(X)  # Add intercept term
beta = [1, 2, 3]
y = np.dot(X, beta) + np.random.normal(0, 1, 100)

# Fit OLS model
model = sm.OLS(y, X)
results = model.fit()

# Print summary of the model
print(results.summary())
