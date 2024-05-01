"""
27. Pipeline Construction: Building end-to-end workflows for data preprocessing, feature engineering, and model training.

A pipeline in scikit-learn allows you to chain together multiple data processing steps into a single workflow. 
This is particularly useful for building end-to-end machine learning pipelines that include data preprocessing, feature engineering, 
and model training. Pipelines ensure that each step in the workflow is executed in the correct order and that transformations 
are applied consistently to both the training and testing data.

Explanation:

We import necessary libraries including load_iris from sklearn.datasets, train_test_split from sklearn.model_selection, StandardScaler, PCA, SVC, 
and Pipeline from sklearn.preprocessing, sklearn.decomposition, sklearn.svm, and sklearn.pipeline, respectively.

We load the Iris dataset, which is a commonly used dataset for classification tasks.

We split the data into training and testing sets using the train_test_split function.

We create a pipeline using the Pipeline class. The pipeline consists of three steps:

StandardScaler: Standardizes features by removing the mean and scaling to unit variance.

PCA: Performs Principal Component Analysis to reduce the dimensionality of the data to 2 components.

SVC: Support Vector Classifier for classification.

We fit the pipeline to the training data, which applies each step in the pipeline to the training data sequentially.

Finally, we evaluate the pipeline's performance on the test set using the score method, which calculates the accuracy of the model on the test data.

"""

# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a pipeline with three steps: StandardScaler, PCA, and SVC
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('classifier', SVC())
])

# Fitting the pipeline to the training data
pipeline.fit(X_train, y_train)

# Evaluating the pipeline on the test set
test_score = pipeline.score(X_test, y_test)
print("Test set accuracy:", test_score)


