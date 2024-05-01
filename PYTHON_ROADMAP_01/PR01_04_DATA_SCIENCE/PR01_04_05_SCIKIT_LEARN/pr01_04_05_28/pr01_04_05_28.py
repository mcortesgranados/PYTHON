"""
28. Model Persistence: Saving trained models to disk for later use.

Model persistence refers to the ability to save trained machine learning models to disk so that they can be reused or deployed later 
without the need to retrain them. This is a crucial aspect of the machine learning workflow, especially when working on larger datasets 
or computationally intensive models. In scikit-learn, model persistence can be achieved using the joblib library or Python's built-in pickle module.

Let's see how to save and load a trained model using scikit-learn and joblib:

Explanation:

We import necessary libraries including load_iris from sklearn.datasets, train_test_split from sklearn.model_selection, 
RandomForestClassifier from sklearn.ensemble, accuracy_score from sklearn.metrics, and dump and load from joblib.

We load the Iris dataset, split it into training and testing sets, and perform the standard train-test split.

We create a Random Forest classifier and train it on the training data.

We save the trained model to disk using the dump function from joblib. The first argument to dump is the trained model object, and 
the second argument is the filename where the model will be saved.

We load the saved model from disk using the load function from joblib.

We make predictions on the test set using the loaded model.

Finally, we evaluate the performance of the loaded model by calculating the accuracy score on the test set.

Model persistence allows us to save trained models and use them in future applications or share them with others without the need to 
retrain them every time. This is particularly useful in production environments where trained models need to be deployed for making predictions on new data.

"""

# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Saving the trained model to disk
dump(clf, 'random_forest_model.joblib')

# Loading the saved model from disk
loaded_model = load('random_forest_model.joblib')

# Making predictions using the loaded model
y_pred = loaded_model.predict(X_test)

# Evaluating the performance of the loaded model
accuracy = accuracy_score(y_test, y_pred)
print("Test set accuracy of the loaded model:", accuracy)
