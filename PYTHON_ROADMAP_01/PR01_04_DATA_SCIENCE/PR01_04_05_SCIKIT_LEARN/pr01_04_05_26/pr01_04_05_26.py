"""
26. Grid Search: Exhaustively searching for the best combination of hyperparameters for a model.

Grid search is a technique used to find the best combination of hyperparameters for a machine learning model by exhaustively 
searching through a specified grid of parameter values. This is particularly useful when training a model with multiple 
hyperparameters, as it allows us to find the combination that yields the best performance.

Explanation:

We import necessary libraries including load_iris from sklearn.datasets, GridSearchCV and train_test_split from sklearn.model_selection, and SVC from sklearn.svm.

We load the Iris dataset, which is a commonly used dataset for classification tasks.

We split the data into training and testing sets using the train_test_split function.

We create an instance of the Support Vector Classifier (SVC) model.

We define a grid of hyperparameters to search using a dictionary param_grid. In this example, we specify different values for the regularization parameter C, 
the kernel type (linear or rbf), and the kernel coefficient gamma.

We perform grid search with cross-validation (cv=5) using the GridSearchCV class. The n_jobs=-1 parameter allows grid search to use all available 
CPU cores for parallel computation.

We fit the grid search object to the training data, which exhaustively searches through the specified grid of hyperparameters.

We print the best hyperparameters found by grid search using the best_params_ attribute.

Finally, we evaluate the model's performance on the test set using the best hyperparameters found and print the test set accuracy.

"""

# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a Support Vector Classifier (SVC) model
svc = SVC()

# Defining a grid of hyperparameters to search
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': [0.1, 0.01, 0.001]}

# Performing grid search with cross-validation
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Printing the best hyperparameters found
print("Best hyperparameters:", grid_search.best_params_)

# Evaluating the model on the test set
test_score = grid_search.score(X_test, y_test)
print("Test set accuracy:", test_score)
