# File name: 959_Logistic Regression.py
# @author Manuela Cortes Granados - 15 Abril 2024 2:48 AM
"""
Logistic Regression is a widely used statistical technique for binary classification. It models the probability that a
 given input belongs to a particular class. Despite its name, logistic regression is used for classification rather than regression.

In this example:

1. We generate synthetic data for binary classification using make_classification from scikit-learn.

2. We split the data into training and testing sets using train_test_split.

3. We initialize and fit a logistic regression model using LogisticRegression from scikit-learn.

4. We predict probabilities for each class using predict_proba and predict classes using predict.

5. Finally, we calculate the accuracy of the model using accuracy_score from scikit-learn.

This example demonstrates how to perform Logistic Regression for binary classification using scikit-learn.


"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate synthetic data for binary classification
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predict probabilities and classes
probabilities = log_reg.predict_proba(X_test)
predicted_classes = log_reg.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predicted_classes)
print(f"Accuracy: {accuracy:.2f}")
