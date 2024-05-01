"""
08. Hyperparameter Tuning: Optimizing model parameters to improve performance.

Explanation:

Importing Libraries: We import necessary libraries including scikit-learn for machine learning algorithms and datasets, and relevant modules 
for model selection and evaluation.

Loading the Dataset: We load the Iris dataset from scikit-learn's datasets module, which is a classic dataset often used for classification tasks.

Splitting the Data: We split the dataset into training and testing sets using train_test_split from scikit-learn's model_selection module.

Defining the Classifier: We define the RandomForestClassifier which we'll use as the base estimator for hyperparameter tuning.

Defining Hyperparameters to Tune: We define a grid of hyperparameters to search over using GridSearchCV. In this example, we vary the number of 
estimators, maximum depth, minimum samples split, and minimum samples leaf.

Instantiating GridSearchCV: We instantiate GridSearchCV with the defined estimator, parameter grid, cross-validation strategy (cv), 
number of parallel jobs (n_jobs), and scoring metric.

Fitting the Grid Search to the Data: We fit the grid search object to the training data to search for the best combination of hyperparameters.

Getting the Best Parameters and Best Score: We retrieve the best parameters and best score found during the grid search.

Making Predictions: We make predictions on the test set using the best model obtained from the grid search.

Calculating Accuracy: We calculate the accuracy of the best model on the test set.

Printing Results: We print the best parameters, best score, and accuracy on the test set to evaluate the performance of the tuned model.
"""

# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the classifier
clf = RandomForestClassifier(random_state=42)

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Instantiate GridSearchCV
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Make predictions on the test set using the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print results
print("Best Parameters:", best_params)
print("Best Score:", best_score)
print("Accuracy on Test Set:", accuracy)
