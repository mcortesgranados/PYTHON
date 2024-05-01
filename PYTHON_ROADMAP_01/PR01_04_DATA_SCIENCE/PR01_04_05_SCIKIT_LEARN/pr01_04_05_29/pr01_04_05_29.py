"""
29. Data Preprocessing: Transforming raw data into a format suitable for modeling.

Data preprocessing is a crucial step in the machine learning pipeline, where raw data is transformed and prepared for modeling. 
This includes tasks such as handling missing values, scaling features, encoding categorical variables, and splitting the data into 
training and testing sets. In scikit-learn, data preprocessing can be accomplished using various preprocessing techniques available 
in the sklearn.preprocessing module.

Let's see an example of data preprocessing including handling missing values, scaling features, and encoding categorical variables:

Explanation:

We import necessary libraries including numpy, pandas, and various modules from sklearn.preprocessing.

We load the Iris dataset and convert it to a pandas DataFrame for easier manipulation.

We introduce missing values in one of the features of the dataset to simulate real-world scenarios.

We split the data into features (X) and the target variable (y).

We handle missing values using SimpleImputer from sklearn.impute, which replaces missing values with the mean of the feature.

We split the data into training and testing sets using train_test_split from sklearn.model_selection.

We scale the features using StandardScaler from sklearn.preprocessing, which standardizes the features by removing the mean and scaling to unit variance.

We encode the categorical target variable using OneHotEncoder from sklearn.preprocessing, which converts categorical integer features into one-hot encoded 
features.

Finally, we print the shape of the preprocessed data to verify the transformations.

Data preprocessing is essential for building accurate and robust machine learning models, and scikit-learn provides a wide range of tools to 
facilitate this process.

"""

# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(data=X, columns=iris.feature_names)
df['target'] = y

# Introducing missing values
df.loc[::20, 'sepal length (cm)'] = np.nan

# Splitting the data into features and target variable
X = df.drop('target', axis=1)
y = df['target']

# Handling missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Scaling features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Encoding categorical variables using OneHotEncoder
encoder = OneHotEncoder()
y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.values.reshape(-1, 1))

# Print the shape of preprocessed data
print("Shape of X_train_scaled:", X_train_scaled.shape)
print("Shape of X_test_scaled:", X_test_scaled.shape)
print("Shape of y_train_encoded:", y_train_encoded.shape)
print("Shape of y_test_encoded:", y_test_encoded.shape)
