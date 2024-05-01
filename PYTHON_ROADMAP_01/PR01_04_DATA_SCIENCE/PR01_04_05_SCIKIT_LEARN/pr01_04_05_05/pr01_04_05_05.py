"""
05. Anomaly Detection: Identifying outliers or unusual patterns in data.

Explanation:

Importing Libraries: We import necessary libraries including numpy for numerical computations, matplotlib for visualization, and scikit-learn 
for anomaly detection.

Generating Synthetic Data: We generate synthetic data using make_blobs from scikit-learn's datasets module. We intentionally introduce outliers in the data by adding random points outside the main cluster.

Initializing Isolation Forest Model: We initialize an Isolation Forest model, which is a popular algorithm for anomaly detection.

Fitting the Model: We fit the Isolation Forest model to the data.

Predicting Outliers: We predict outliers/anomalies in the data using the fitted model.

Visualizing Data and Outliers: We visualize the data and outliers using matplotlib. Points classified as outliers are shown in a different color, making it easier to identify them.
"""

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest

# Generating synthetic data with outliers
X, _ = make_blobs(n_samples=1000, centers=1, random_state=42)
outliers = np.random.uniform(low=-10, high=10, size=(50, 2))
X = np.vstack([X, outliers])

# Initialize Isolation Forest model
isolation_forest = IsolationForest(contamination=0.05, random_state=42)

# Fit the model to the data
isolation_forest.fit(X)

# Predict outliers/anomalies
outlier_preds = isolation_forest.predict(X)

# Visualize the data and outliers
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=outlier_preds, cmap='viridis', edgecolor='k', s=50)
plt.title('Anomaly Detection with Isolation Forest')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Anomaly Score')
plt.show()
