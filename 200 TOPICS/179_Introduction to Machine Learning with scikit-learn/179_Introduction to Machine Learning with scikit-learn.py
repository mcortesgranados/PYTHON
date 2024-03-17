# In this script:

# We import necessary libraries including NumPy for numerical computations, scikit-learn for machine learning functionalities, and matplotlib for data
#  visualization (not used in this example).
# We generate some synthetic data using NumPy's random number generator.
# We split the data into training and testing sets using train_test_split function from scikit-learn.
# We create a Linear Regression model and train it using the training data.
# We make predictions on the test set using the trained model.
# We evaluate the model's performance using Mean Squared Error (MSE) metric.
# This is just a basic example to introduce you to the usage of scikit-learn for machine learning tasks. You can explore more advanced techniques and 
# models provided by scikit-learn for various machine learning tasks.


# The Mean Squared Error (MSE) is a metric used to evaluate the performance of a regression model. It measures the average squared difference between 
# the actual values (y_true) and the predicted values (y_pred).

# In your specific case, a Mean Squared Error of approximately 0.9177 means that, on average, the squared difference between the actual target 
# values and the predicted values is approximately 0.9177.

# Since MSE is a measure of the model's accuracy, lower values of MSE indicate better performance. Therefore, a MSE of 0.9177 suggests that the 
# model's predictions are relatively close to the actual values, but there is still room for improvement.

# Importing necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generating some synthetic data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2.5 * X.squeeze() + np.random.randn(100)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

