"""
01. Regression: Predicting continuous target variables.

Explanation:

Importing Libraries: We import the necessary libraries including numpy for numerical computations, scikit-learn for classification modeling, 
and matplotlib for visualization.

Generating Data: We generate synthetic classification data using the make_classification function from scikit-learn. This function creates a 
dataset with specified number of samples, features, classes, and random state.

Splitting Data: We split the generated data into training and testing sets using the train_test_split function. 
This allows us to train the model on a subset of the data and evaluate its performance on unseen data.

Initializing Model: We initialize a logistic regression model using the LogisticRegression class from scikit-learn.

Training Model: We train the logistic regression model on the training data using the fit method.

Making Predictions: We use the trained model to make predictions on the testing data using the predict method.

Evaluating Model: We compute evaluation metrics such as accuracy, classification report, and confusion matrix to assess the performance of the model on the testing data.

Printing Results: We print out the evaluation metrics to evaluate the performance of the classification model.

"""

# Importing necessary libraries
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate synthetic regression data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Compute evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Visualize the regression line
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.title('Regression: Actual vs Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
