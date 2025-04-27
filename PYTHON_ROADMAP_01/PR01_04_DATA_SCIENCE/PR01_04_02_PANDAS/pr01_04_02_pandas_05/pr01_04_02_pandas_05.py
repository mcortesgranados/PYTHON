"""
05. Handling missing or null values using functions like isnull(), fillna(), dropna(), etc.
"""

import pandas as pd
import numpy as np

# Sample data: Create a DataFrame with missing values
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', np.nan],
    'Age': [24, 27, np.nan, 32, 29],
    'City': ['New York', np.nan, 'Chicago', 'Houston', 'Phoenix'],
    'Salary': [50000, 60000, 55000, np.nan, 65000]
}

# Create DataFrame
df = pd.DataFrame(data)

# 1. Checking for missing values using isnull()
print("Checking for missing values using isnull():")
print(df.isnull())

# 2. Counting missing values in each column
print("\nCounting missing values in each column:")
print(df.isnull().sum())

# 3. Filling missing values using fillna()
# Fill missing 'Age' values with the mean of 'Age'
mean_age = df['Age'].mean()
df_filled = df.fillna({'Age': mean_age, 'Salary': 60000, 'City': 'Unknown', 'Name': 'Unknown'})
print("\nFilling missing values using fillna():")
print(df_filled)

# 4. Dropping rows with missing values using dropna()
# Drop rows that contain any missing values
df_dropped = df.dropna()
print("\nDropping rows with missing values using dropna():")
print(df_dropped)

# 5. Dropping columns with missing values using dropna()
# Drop columns that contain any missing values
df_dropped_columns = df.dropna(axis=1)
print("\nDropping columns with missing values using dropna() (axis=1):")
print(df_dropped_columns)

# 6. Filling missing values with forward fill method using fillna()
# Forward fill to propagate the last valid value
df_forward_filled = df.fillna(method='ffill')
print("\nFilling missing values using forward fill (ffill):")
print(df_forward_filled)

# 7. Filling missing values with backward fill method using fillna()
# Backward fill to propagate the next valid value
df_backward_filled = df.fillna(method='bfill')
print("\nFilling missing values using backward fill (bfill):")
print(df_backward_filled)
