import pandas as pd

# Read data from CSV file
df = pd.read_csv('data_large.csv')

# Perform GroupBy operation and calculate mean of 'Value' column
grouped_data = df.groupby('Category')['Value'].mean()

# Display the grouped data
print(grouped_data)
