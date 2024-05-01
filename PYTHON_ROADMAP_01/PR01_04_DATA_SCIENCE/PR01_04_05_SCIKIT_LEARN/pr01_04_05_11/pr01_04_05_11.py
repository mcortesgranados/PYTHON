"""
11. Imbalanced Data Handling: Dealing with datasets where one class is significantly more prevalent than others.

Dealing with imbalanced data is a common challenge in machine learning, especially in classification tasks where one class is significantly 
more prevalent than others. Scikit-learn provides various techniques to handle imbalanced data, such as resampling methods, 
class weights, and ensemble methods. Here's a Python example demonstrating how to use the Random Forest classifier with class 
weights to handle imbalanced data:

Explanation:

Importing Libraries: We import necessary libraries including scikit-learn for machine learning algorithms and datasets.

Generating Imbalanced Data: We generate imbalanced synthetic data using the make_classification function from scikit-learn's datasets module. 
We specify the weights parameter to make one class (minority class) significantly less prevalent than the other class (majority class).

Splitting the Dataset: We split the generated dataset into training and testing sets using the train_test_split function from scikit-learn's 
model_selection module.

Defining the Classifier with Class Weights: We define the RandomForestClassifier with class_weight='balanced', which automatically 
adjusts the class weights inversely proportional to class frequencies in the input data.

Training the Classifier: We train the Random Forest classifier on the training data using the fit method.

Making Predictions: We make predictions on the testing data using the trained classifier.

Printing Classification Report: We print the classification report, which includes metrics such as precision, recall, and F1-score, 
to evaluate the performance of the model on the imbalanced data.

"""

# Importing necessary libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Generate imbalanced data
X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                            n_informative=3, n_redundant=1, flip_y=0,
                            n_features=20, n_clusters_per_class=1,
                            n_samples=1000, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Random Forest classifier with class weights
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


