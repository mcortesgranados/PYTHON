"""
14. Performing statistical analysis and hypothesis testing using functions like corr(), cov(), ttest_ind(), etc.

"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# Create a sample DataFrame
data = {
    'A': np.random.normal(loc=0, scale=1, size=100),
    'B': np.random.normal(loc=0, scale=1, size=100),
    'C': np.random.normal(loc=0, scale=1, size=100)
}
df = pd.DataFrame(data)

# Display the original DataFrame
print("Original DataFrame:")
print(df)

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Display the correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Calculate the covariance matrix
covariance_matrix = df.cov()

# Display the covariance matrix
print("\nCovariance Matrix:")
print(covariance_matrix)

# Perform a two-sample t-test for the equality of means of two independent samples (columns A and B)
t_statistic, p_value = ttest_ind(df['A'], df['B'])

# Display the results of the t-test
print("\nResults of t-test:")
print("T-statistic:", t_statistic)
print("P-value:", p_value)
