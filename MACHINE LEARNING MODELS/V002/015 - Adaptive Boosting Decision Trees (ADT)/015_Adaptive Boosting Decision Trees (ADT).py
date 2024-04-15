# File name: 015_Adaptive Boosting Decision Trees (ADT).py
# @author Manuela Cortes Granados - 14 Abril 2024 1:42 PM

"""
This code snippet demonstrates how to use the AdaBoost classifier (AdaBoostClassifier) in scikit-learn with decision trees (DecisionTreeClassifier) 
as weak learners. Here's a breakdown of the steps:

Load Dataset: Load the Iris dataset using load_iris() from scikit-learn. This dataset contains features of iris flowers and their corresponding
 species labels.

Split Dataset: Split the dataset into training and testing sets using train_test_split() to evaluate the performance of the model.

Define Base Classifier: Create a base decision tree classifier (DecisionTreeClassifier) with a maximum depth of 1. This decision tree
 will serve as the weak learner in AdaBoost.

Initialize AdaBoost Classifier: Create an AdaBoost classifier (AdaBoostClassifier) and specify the base estimator (the base decision tree classifier)
 and the number of weak learners (n_estimators).

Train the Model: Fit the AdaBoost classifier to the training data using the fit() method.

Make Predictions: Use the trained model to make predictions on the testing set with the predict() method.

Evaluate Model: Calculate the accuracy of the model on the testing set using accuracy_score().

You can replace the Iris dataset with any other dataset of your choice and adjust the parameters of the AdaBoost classifier as needed for
 your specific problem.
"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the AdaBoost classifier with 50 weak learners
adb_classifier = AdaBoostClassifier(n_estimators=50, random_state=42)

# Train the AdaBoost classifier
adb_classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = adb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

