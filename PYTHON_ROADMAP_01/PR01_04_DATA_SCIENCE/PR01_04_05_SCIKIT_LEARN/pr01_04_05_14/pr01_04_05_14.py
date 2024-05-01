"""
14. Image Classification: Categorizing images into predefined classes or categories.

Image classification is a task where we categorize images into predefined classes or categories. 
Scikit-learn is not specifically designed for image classification tasks, as it focuses more on traditional 
machine learning algorithms rather than deep learning. However, we can still demonstrate a basic example of image classification 
using scikit-learn by converting images into feature vectors and then applying a machine learning algorithm.

Explanation:

Loading the MNIST Dataset: We load the MNIST dataset using scikit-learn's fetch_openml function. 
The dataset contains 28x28 pixel images of handwritten digits (0-9).

Splitting the Dataset: We split the dataset into training and testing sets using the train_test_split function.

Preprocessing the Data: We standardize the features by scaling them using StandardScaler.

Training a Support Vector Machine (SVM) Classifier: We train an SVM classifier using the radial basis function (RBF) kernel.

Making Predictions: We make predictions on the testing set using the trained SVM classifier.

Evaluating the Model: We evaluate the model's performance by calculating accuracy and generating a classification report 
using scikit-learn's accuracy_score and classification_report functions.

"""

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Loading the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training a support vector machine (SVM) classifier
svm_clf = SVC(kernel='rbf', gamma='scale', random_state=42)
svm_clf.fit(X_train_scaled, y_train)

# Making predictions
y_pred = svm_clf.predict(X_test_scaled)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Displaying the classification report
print("Classification Report:")
print(classification_rep)

# Displaying the accuracy
print("Accuracy:", accuracy)
