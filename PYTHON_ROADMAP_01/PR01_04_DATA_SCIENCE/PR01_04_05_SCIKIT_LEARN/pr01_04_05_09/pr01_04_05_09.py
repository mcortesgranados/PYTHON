"""
09. Cross-Validation: Evaluating model performance using different subsets of data.

Explanation:

Importing Libraries: We import necessary libraries including scikit-learn for machine learning algorithms and datasets.

Loading the Dataset: We load the Iris dataset from scikit-learn's datasets module, which is a classic dataset often used for classification tasks.

Defining the Classifier: We define the DecisionTreeClassifier which we'll use as the classifier for cross-validation.

Performing Cross-Validation: We use the cross_val_score function from scikit-learn's model_selection module to perform 5-fold cross-validation. It splits the data into 5 equal parts, trains the model on 4 parts, and evaluates it on the remaining part. This process is repeated 5 times.

Printing Cross-Validation Scores: We print the cross-validation scores obtained for each fold.

Calculating Mean Accuracy: We calculate the mean accuracy across all folds and print it as the overall performance of the model.

"""

# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Define the classifier
clf = DecisionTreeClassifier(random_state=42)

# Perform 5-fold cross-validation
scores = cross_val_score(clf, X, y, cv=5)

# Print the cross-validation scores
print("Cross-Validation Scores:", scores)

# Calculate and print the mean accuracy
mean_accuracy = scores.mean()
print("Mean Accuracy:", mean_accuracy)


