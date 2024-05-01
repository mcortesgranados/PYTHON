"""
22. Density Estimation: Estimating the probability density function of a random variable.

Density estimation is a fundamental task in statistics and machine learning, which involves estimating the probability density function (PDF) 
of a random variable based on a sample of data points. Kernel Density Estimation (KDE) is a popular technique for density estimation. 
Here's a Python example using scikit-learn to perform KDE:

Explanation:

We import necessary libraries including NumPy, Matplotlib, and scikit-learn's KernelDensity algorithm.

We generate synthetic data by concatenating two normal distributions with different means.

We visualize the synthetic data using Matplotlib's hist function to show the histogram of data points.

We instantiate a KernelDensity object with the desired kernel (Gaussian) and bandwidth.

We fit the KDE model to the data using the fit method.

We generate new data points for plotting the estimated density function.

We compute the log density estimates for the new data points using the score_samples method.

We visualize the estimated density function by filling the area under the curve using Matplotlib's fill_between function.

"""

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Generating synthetic data
np.random.seed(0)
X = np.concatenate([np.random.normal(0, 1, 1000), np.random.normal(4, 1, 1000)])[:, np.newaxis]

# Visualizing the data
plt.hist(X, bins=50, density=True, alpha=0.5, color='blue')
plt.xlabel('Feature')
plt.ylabel('Density')
plt.title('Synthetic Data')
plt.show()

# Instantiating and fitting the Kernel Density Estimation model
kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
kde.fit(X)

# Generating new data points for plotting the estimated density function
X_new = np.linspace(-5, 10, 1000)[:, np.newaxis]
log_dens = kde.score_samples(X_new)

# Visualizing the estimated density function
plt.fill_between(X_new[:, 0], np.exp(log_dens), alpha=0.5, color='red')
plt.xlabel('Feature')
plt.ylabel('Density')
plt.title('Kernel Density Estimation')
plt.show()


