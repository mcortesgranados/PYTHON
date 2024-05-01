"""
23. Outlier Detection: Identifying unusual observations that deviate from normal behavior.

Outlier detection is a technique used to identify observations in a dataset that significantly deviate from the rest of the data points. 
These outliers may represent erroneous data, anomalies, or rare events. One common method for outlier detection is using the Isolation 
Forest algorithm, which isolates outliers by randomly partitioning the data into subsets. Here's a Python example using scikit-learn to 
perform outlier detection using Isolation Forest:

xplanation:

We import necessary libraries including NumPy, Matplotlib, and scikit-learn's IsolationForest algorithm.
We generate synthetic data containing normal observations and outliers using NumPy.
We visualize the synthetic data using Matplotlib's scatter function to create a scatter plot.
We instantiate an IsolationForest object with the desired contamination parameter and random state.
We fit the Isolation Forest model to the data using the fit method.
We predict outliers in the dataset using the predict method.
We visualize the outliers by coloring them differently in the scatter plot.

"""

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Generating synthetic data
np.random.seed(0)
X = 0.3 * np.random.randn(100, 2)
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.vstack([X, X_outliers])

# Visualizing the synthetic data
plt.scatter(X[:, 0], X[:, 1], c='blue', edgecolors='k', alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Synthetic Data')
plt.show()

# Instantiating and fitting the Isolation Forest model
clf = IsolationForest(contamination=0.1, random_state=0)
clf.fit(X)

# Predicting outliers
y_pred = clf.predict(X)

# Visualizing the outliers
plt.scatter(X[:, 0], X[:, 1], c=y_pred, edgecolors='k', alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Outlier Detection using Isolation Forest')
plt.show()
