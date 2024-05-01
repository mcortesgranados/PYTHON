"""
07. Model Evaluation: Assessing the performance of machine learning models using metrics like accuracy, precision, recall, F1-score, etc.

Explanation:

Importing Libraries: We import necessary libraries including scikit-learn for machine learning algorithms and datasets, and relevant modules for model evaluation.

Loading the Dataset: We load the Iris dataset from scikit-learn's datasets module, which is a classic dataset often used for classification tasks.

Splitting the Data: We split the dataset into training and testing sets using train_test_split from scikit-learn's model_selection module.

Training a Classifier: We train a RandomForestClassifier on the training data.

Making Predictions: We make predictions on the testing data.

Calculating Evaluation Metrics: We calculate evaluation metrics such as accuracy, precision, recall, and F1-score using appropriate functions 
from scikit-learn's metrics module.

Printing Evaluation Metrics: We print the calculated evaluation metrics.

Printing Classification Report: We print a classification report containing precision, recall, F1-score, and support for each class.

Printing Confusion Matrix: We print the confusion matrix to evaluate the performance of the classifier.

"""

# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print evaluation metrics
print("Evaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
