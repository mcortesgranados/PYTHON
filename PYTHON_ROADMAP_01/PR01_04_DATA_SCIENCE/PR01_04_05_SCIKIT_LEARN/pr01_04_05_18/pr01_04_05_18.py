"""
18. Model Interpretation: Understanding and interpreting the decisions made by machine learning models.

Model interpretation involves understanding and interpreting the decisions made by machine learning models to gain insights 
into how they work and why they make certain predictions. In scikit-learn, we can use various techniques to interpret models, 
such as feature importance, partial dependence plots, and model-specific methods. Here's a Python example demonstrating model interpretation using scikit-learn:

Explanation:

Import Libraries: We import necessary classes and functions from scikit-learn, including RandomForestClassifier for creating a random forest model and 
plot_partial_dependence for generating partial dependence plots.

Load Dataset: We load the Iris dataset, which is a popular dataset for classification tasks.

Split Data: We split the dataset into training and testing sets using train_test_split.

Create and Train Model: We create a Random Forest classifier and train it on the training data.

Make Predictions: We make predictions on the test set using the trained model.

Calculate Accuracy: We calculate the accuracy of the model on the test set.

Feature Importance: We print the feature importance scores calculated by the random forest model. Feature importance indicates the 
contribution of each feature to the model's predictions.

Partial Dependence Plot: We generate a partial dependence plot for the first feature (sepal length) to visualize the relationship 
between this feature and the target variable while marginalizing over the other features.

"""

# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Feature importance
print("Feature Importance:")
feature_importance = clf.feature_importances_
for i, importance in enumerate(feature_importance):
    print(f"Feature {i+1}: {importance}")

# Partial dependence plot (for the first feature)
import matplotlib.pyplot as plt
from sklearn.inspection import plot_partial_dependence

print("Partial Dependence Plot:")
fig, ax = plt.subplots()
plot_partial_dependence(clf, X_train, features=[0], ax=ax)
plt.xlabel(iris.feature_names[0])
plt.ylabel("Partial Dependence")
plt.show()
