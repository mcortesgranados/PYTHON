""""
02. Exploring and inspecting data using functions like head(), tail(), info(), describe(), etc.
"""

import pandas as pd

# Sample data: Create a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [24, 27, 22, 32, 29],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'Salary': [50000, 60000, 55000, 70000, 65000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Explore the data using pandas functions

# 1. head(): Shows the first few rows (default 5)
print("First 5 rows:")
print(df.head())

# 2. tail(): Shows the last few rows (default 5)
print("\nLast 5 rows:")
print(df.tail())

# 3. info(): Provides a concise summary of the DataFrame
print("\nDataFrame Info:")
print(df.info())

# 4. describe(): Provides summary statistics for numerical columns
print("\nSummary Statistics:")
print(df.describe())
