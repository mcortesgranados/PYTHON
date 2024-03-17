import pandas as pd

# Read data from CSV file
df = pd.read_csv('data.csv')

# Display first few rows
print(df.head())

# Get basic information about the DataFrame
print(df.info())

# Check for missing values
print(df.isnull().sum())
