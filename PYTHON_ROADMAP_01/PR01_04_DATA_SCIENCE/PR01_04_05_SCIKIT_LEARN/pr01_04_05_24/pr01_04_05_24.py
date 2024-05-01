"""
01. Regression: Predicting continuous target variables.

Regression is a supervised learning technique used to predict continuous target variables based on one or more input features. 
It's widely used in various domains such as finance, healthcare, and engineering. Here's a Python example using scikit-learn to perform linear regression:

Explanation:

We import necessary libraries including NumPy, Matplotlib, scikit-learn's LinearRegression, train_test_split, and mean_squared_error.

We generate synthetic data with a linear relationship between the feature (X) and target (y).

We visualize the synthetic data using Matplotlib's scatter function to create a scatter plot.

We split the data into training and testing sets using train_test_split.

We instantiate a LinearRegression object and fit it to the training data using the fit method.

We make predictions on the test set using the predict method.

We calculate the Mean Squared Error (MSE) between the actual and predicted values using mean_squared_error.

We visualize the regression line along with the actual and predicted values.

Finally, we print the Mean Squared Error, which quantifies the model's performance.

"""

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generating synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # Feature (input variable)
y = 4 + 3 * X + np.random.randn(100, 1)  # Target (output variable)

# Visualizing the synthetic data
plt.scatter(X, y, color='blue', alpha=0.6)
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Synthetic Data')
plt.show()

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiating and fitting the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Calculating Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Visualizing the regression line
plt.scatter(X_test, y_test, color='blue', alpha=0.6, label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Linear Regression')
plt.legend()
plt.show()

# Printing the Mean Squared Error
print("Mean Squared Error:", mse)
