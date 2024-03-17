# In this example:

# We generate random data X and y.
# We add a constant to the independent variable X to include the intercept term in the regression model.
# We fit the linear regression model using sm.OLS() (ordinary least squares) from statsmodels.
# We print the summary of the regression results which includes coefficients, standard errors, p-values, etc.
# We make predictions using the fitted model.

import numpy as np
import statsmodels.api as sm

# Generate some random data for demonstration
np.random.seed(0)
X = np.random.rand(100, 1)  # Independent variable
y = 2 * X.squeeze() + np.random.randn(100)  # Dependent variable

# Add a constant to the independent variable for the intercept term
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the summary of the regression results
print(model.summary())

# Predictions
predictions = model.predict(X)
print(predictions)
