"""
06. Feature Selection: Selecting the most relevant features for modeling.

Explanation:

Importing Libraries: We import necessary libraries including numpy for numerical computations, scikit-learn for machine learning algorithms and
datasets, and relevant modules for feature selection and model evaluation.

Loading the Dataset: We load the Iris dataset from scikit-learn's datasets module, which is a classic dataset often used for classification tasks.

Splitting the Data: We split the dataset into training and testing sets using train_test_split from scikit-learn's model_selection module.

Initializing SelectKBest: We initialize SelectKBest, which is a univariate feature selection method, with ANOVA F-value scoring function.

Fitting SelectKBest: We fit SelectKBest to the training data to select the top k=2 most relevant features based on ANOVA F-value.

Printing Selected Features: We print the names of the selected features.

Training a Classifier: We train a RandomForestClassifier on the selected features.

Transforming Testing Data: We transform the testing data with the same selected features.

Making Predictions: We make predictions on the testing data.
Calculating Accuracy: We calculate the accuracy of the classifier on the testing data.

"""

# Importing necessary libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SelectKBest with ANOVA F-value scoring
k_best = SelectKBest(score_func=f_classif, k=2)

# Fit SelectKBest to the training data
X_train_kbest = k_best.fit_transform(X_train, y_train)

# Print the selected features
print("Selected Features after SelectKBest:")
print(iris.feature_names[k_best.get_support()])

# Train a classifier on the selected features
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_kbest, y_train)

# Transform the testing data with the same selected features
X_test_kbest = k_best.transform(X_test)

# Make predictions on the testing data
y_pred = clf.predict(X_test_kbest)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy:.2f}")
