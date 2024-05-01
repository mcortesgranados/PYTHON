"""
13. Time Series Forecasting: Predicting future values based on historical time-series data.

Explanation:

Generating Synthetic Time-Series Data: We generate synthetic time-series data using NumPy's arange function to create a sequence of numbers as the 
time index and adding random noise to a sinusoidal signal to create the target variable.

Splitting the Data: We split the generated data into training and testing sets using scikit-learn's train_test_split function.

Training a Linear Regression Model: We train a linear regression model using the training data.

Making Predictions: We make predictions on both the training and testing data using the trained linear regression model.

Evaluating the Model: We calculate the Root Mean Squared Error (RMSE) to evaluate the performance of the model on both the training and testing data.

Plotting the Results: We plot the original time-series data points, along with the predicted values from the linear regression model for both the 
training and testing sets.

"""

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generating synthetic time-series data
np.random.seed(42)
n_samples = 100
X = np.arange(n_samples).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(scale=0.1, size=n_samples)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluating the model
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Train Data')
plt.scatter(X_test, y_test, color='red', label='Test Data')
plt.plot(X_train, y_pred_train, color='green', label=f'Train Predictions (RMSE: {train_rmse:.2f})')
plt.plot(X_test, y_pred_test, color='orange', label=f'Test Predictions (RMSE: {test_rmse:.2f})')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Time Series Forecasting with Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
