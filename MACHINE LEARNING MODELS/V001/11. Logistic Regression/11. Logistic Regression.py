
# Logistic Regression is a statistical method used for binary classification tasks, where the target variable (or outcome) is categorical 
# and has only two possible outcomes. It models the probability that a given input belongs to a particular category.

# Here's a simple example of how to implement logistic regression using Python's scikit-learn library:

# In this example:

# We load the Iris dataset, which is a well-known dataset in machine learning with three classes of iris plants.
# We split the dataset into training and testing sets using the train_test_split function.
# We initialize a logistic regression model using LogisticRegression.
# We train the model on the training data using the fit method.
# We use the trained model to make predictions on the test data using the predict method.
# Finally, we evaluate the performance of the model by calculating the accuracy of its predictions using the accuracy_score function 
# from scikit-learn's metrics module.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
