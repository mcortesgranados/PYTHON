"""
Explanation:

We import the NumPy library as np.
We create a sample dataset data as a NumPy array.
We perform various statistical operations on the dataset:
np.mean(data): Calculates the mean of the data.
np.median(data): Calculates the median of the data.
np.std(data): Calculates the standard deviation of the data.
np.var(data): Calculates the variance of the data.
We create another dataset data2 as a 2D NumPy array.
We calculate the correlation coefficient using np.corrcoef(data2). This function returns a correlation coefficient matrix where each element (i, j) 
represents the correlation coefficient between the ith and jth variables in the dataset.
Documentation:

np.mean(a, axis=None): Computes the arithmetic mean along the specified axis. If no axis is specified, computes the mean of the flattened array.
np.median(a, axis=None): Computes the median along the specified axis. If no axis is specified, computes the median of the flattened array.
np.std(a, axis=None): Computes the standard deviation along the specified axis. If no axis is specified, computes the standard deviation of the flattened array.
np.var(a, axis=None): Computes the variance along the specified axis. If no axis is specified, computes the variance of the flattened array.
np.corrcoef(x, y=None, rowvar=True): Computes the Pearson correlation coefficients between variables in x. If x is 2D, correlation coefficients 
are computed between columns (default behavior). If rowvar=False, then correlation coefficients are computed between rows.
This example demonstrates how to perform statistical operations such as mean, median, standard deviation, variance, and correlation using NumPy in Python.

"""

import numpy as np

# Create a sample dataset
data = np.array([12, 15, 18, 20, 22, 24, 26, 28, 30])

# Calculate mean
mean_value = np.mean(data)
print("Mean:", mean_value)

# Calculate median
median_value = np.median(data)
print("Median:", median_value)

# Calculate standard deviation
std_deviation = np.std(data)
print("Standard Deviation:", std_deviation)

# Calculate variance
variance = np.var(data)
print("Variance:", variance)

# Create another dataset
data2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Calculate correlation coefficient
correlation_matrix = np.corrcoef(data2)
print("Correlation Coefficient Matrix:")
print(correlation_matrix)
