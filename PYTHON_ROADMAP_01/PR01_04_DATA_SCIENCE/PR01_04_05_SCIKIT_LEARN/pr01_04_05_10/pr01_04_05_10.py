"""
10. Ensemble Learning: Combining multiple models to improve predictive performance.

Ensemble Learning is a powerful technique in machine learning where multiple models are combined to improve predictive performance. 
One common ensemble method is the Random Forest algorithm, which constructs a multitude of decision trees during training and outputs 
the mode of the classes (classification) or the average prediction (regression) of the individual trees.

Explanation:

Importing Libraries: We import necessary libraries including scikit-learn for machine learning algorithms and datasets.

Loading the Dataset: We load the Iris dataset from scikit-learn's datasets module, which is a classic dataset often used for classification tasks.

Splitting the Dataset: We split the dataset into training and testing sets using the train_test_split function from scikit-learn's model_selection module.

Defining the Classifier: We define the RandomForestClassifier which is an ensemble learning method that fits a number of decision tree classifiers 
on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control overfitting.

Training the Classifier: We train the Random Forest classifier on the training data using the fit method.

Making Predictions: We make predictions on the testing data using the trained classifier.

Calculating Accuracy: We calculate the accuracy of the model by comparing the predicted labels with the actual labels using the accuracy_score function 
from scikit-learn's metrics module.

Printing Accuracy: We print the accuracy of the model on the testing data.

"""

# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy:", accuracy)
